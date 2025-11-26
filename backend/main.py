from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from core.apriori import Apriori
from core.kmeans import KMeans
from data.loader import HeartFailureDataLoader
from collections import Counter, defaultdict
import numpy as np
from scipy.sparse import spmatrix
import pandas as pd
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CardioAI API",
    version="2.2",
    description="API d'analyse de données cardiaques avec Apriori et K-Means"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# =====================
# 📦 Modèles Pydantic
# =====================

class AprioriParams(BaseModel):
    min_support: Optional[float] = Field(default=None, ge=0.01, le=1.0)
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)

class KMeansParams(BaseModel):
    n_clusters: int = Field(default=3, ge=2, le=10, description="Nombre de clusters")
    max_iterations: int = Field(default=100, ge=10, le=1000)
    random_state: Optional[int] = Field(default=42)

class AprioriResponse(BaseModel):
    success: bool
    rules: List[Dict[str, Any]]
    attributes: List[str]
    statistics: Dict[str, Any]
    total_rules: int
    total_transactions: int
    execution_time: float
    matrix_type: str

class KMeansResponse(BaseModel):
    success: bool
    clusters: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    labels: List[int]
    centroids: List[List[float]]
    silhouette_score: float
    execution_time: float

class DatasetInfoResponse(BaseModel):
    total_transactions: int
    total_unique_items: int
    avg_transaction_length: float
    top_items: List[Dict[str, Any]]
    categories: Dict[str, int]
    numerical_stats: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str

# =====================
# 🛠️ Utilitaires
# =====================

def to_native(obj):
    """Conversion robuste de tous types NumPy/SciPy en types Python natifs."""
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, spmatrix):
        return obj.toarray().tolist()
    elif isinstance(obj, (list, tuple)):
        return [to_native(i) for i in obj]
    elif isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.integer)):
        return int(obj)
    elif isinstance(obj, (np.floating)):
        return float(obj)
    else:
        return str(obj)

# =====================
# 🚀 Endpoints API
# =====================

@app.post("/run_apriori", response_model=AprioriResponse)
async def run_apriori(params: AprioriParams):
    """
    🔬 Exécute l'algorithme Apriori optimisé
    """
    try:
        logger.info(f"Démarrage Apriori avec params: {params}")
        
        data_loader = HeartFailureDataLoader()
        transactions = data_loader.load_dataset()
        
        if not transactions:
            raise HTTPException(status_code=400, detail="Aucune transaction chargée")

        apriori = Apriori(
            min_support=params.min_support,
            min_confidence=params.min_confidence
        )
        apriori.fit(transactions)
        rules = apriori.generate_rules()
        statistics = apriori.get_statistics()

        rules_native = to_native(rules)
        statistics_native = to_native(statistics)
        
        attributes = sorted({
            str(item) 
            for rule in rules_native 
            for item in rule["antecedent"] + rule["consequent"]
        })

        response = AprioriResponse(
            success=True,
            rules=rules_native,
            attributes=attributes,
            statistics=statistics_native,
            total_rules=len(rules_native),
            total_transactions=len(transactions),
            execution_time=statistics_native['execution_time'],
            matrix_type="Sparse (CSR)" if statistics_native['use_sparse_matrix'] else "Dense (NumPy)"
        )
        
        logger.info(f"Apriori terminé: {len(rules_native)} règles")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur Apriori: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


@app.post("/run_kmeans", response_model=KMeansResponse)
async def run_kmeans(params: KMeansParams):
    """
    🎯 Exécute l'algorithme K-Means sur les données numériques
    """
    try:
        logger.info(f"Démarrage K-Means avec params: {params}")
        
        # Chargement et préparation des données
        data_loader = HeartFailureDataLoader()
        df = data_loader.get_dataframe()
        
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="Dataset vide")
        
        # Sélection des colonnes numériques
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numerical_cols:
            raise HTTPException(status_code=400, detail="Aucune colonne numérique trouvée")
        
        X = df[numerical_cols].values
        
        # Normalisation des données
        X_normalized = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        
        # Exécution K-Means
        kmeans = KMeans(
            n_clusters=params.n_clusters,
            max_iterations=params.max_iterations,
            random_state=params.random_state
        )
        kmeans.fit(X_normalized)
        
        # Calcul du score de silhouette
        silhouette = kmeans.calculate_silhouette_score(X_normalized)
        
        statistics = kmeans.get_statistics()
        
        # Enrichissement des statistiques de cluster
        clusters_info = []
        for i in range(params.n_clusters):
            cluster_mask = kmeans.labels == i
            cluster_data = df[cluster_mask]
            
            clusters_info.append({
                'cluster_id': i,
                'size': int(statistics['cluster_stats'][i]['size']),
                'percentage': float(statistics['cluster_stats'][i]['percentage']),
                'centroid': statistics['cluster_stats'][i]['centroid'],
                'features': {
                    col: {
                        'mean': float(cluster_data[col].mean()),
                        'std': float(cluster_data[col].std()),
                        'min': float(cluster_data[col].min()),
                        'max': float(cluster_data[col].max())
                    }
                    for col in numerical_cols
                }
            })
        
        response = KMeansResponse(
            success=True,
            clusters=clusters_info,
            statistics=to_native(statistics),
            labels=to_native(kmeans.labels),
            centroids=to_native(kmeans.centroids),
            silhouette_score=float(silhouette),
            execution_time=float(statistics['execution_time'])
        )
        
        logger.info(f"K-Means terminé: {params.n_clusters} clusters, silhouette={silhouette:.3f}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur K-Means: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


@app.get("/dataset_info", response_model=DatasetInfoResponse)
async def get_dataset_info():
    """
    📊 Informations détaillées sur le dataset
    """
    try:
        data_loader = HeartFailureDataLoader()
        transactions = data_loader.load_dataset()
        df = data_loader.get_dataframe()
        
        if not transactions:
            raise HTTPException(status_code=400, detail="Dataset vide")
        
        # Analyse des items
        all_items = [item for t in transactions for item in t]
        item_counts = Counter(all_items)
        
        categories: Dict[str, int] = defaultdict(int)
        for item in item_counts.keys():
            category = item.split('_')[0] if '_' in item else "other"
            categories[category] += 1
        
        transaction_lengths = [len(t) for t in transactions]
        avg_len = np.mean(transaction_lengths)
        
        # Statistiques numériques si disponibles
        numerical_stats = None
        if df is not None:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numerical_cols:
                numerical_stats = {
                    col: {
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max())
                    }
                    for col in numerical_cols
                }
        
        response = DatasetInfoResponse(
            total_transactions=len(transactions),
            total_unique_items=len(item_counts),
            avg_transaction_length=float(avg_len),
            top_items=[
                {
                    "item": item,
                    "count": int(count),
                    "percentage": round(float(count) / len(transactions) * 100, 2)
                }
                for item, count in item_counts.most_common(15)
            ],
            categories={k: int(v) for k, v in dict(categories).items()},
            numerical_stats=numerical_stats
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur dataset info: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """🩺 Vérification de la santé du service"""
    return HealthResponse(status="healthy", service="CardioAI API", version="2.2")


@app.get("/")
async def root():
    """📌 Point d'entrée principal"""
    return {
        "service": "CardioAI API",
        "version": "2.2",
        "description": "API d'analyse de données cardiaques",
        "endpoints": {
            "POST /run_apriori": "Algorithme Apriori",
            "POST /run_kmeans": "Algorithme K-Means",
            "GET /dataset_info": "Informations dataset",
            "GET /health": "Santé du service",
            "GET /docs": "Documentation Swagger"
        },
        "algorithms": ["Apriori", "K-Means"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)