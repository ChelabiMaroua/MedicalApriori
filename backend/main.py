from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from core.apriori import Apriori
from data.loader import HeartFailureDataLoader
from collections import Counter, defaultdict
import numpy as np
from scipy.sparse import spmatrix
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CardioAI API",
    version="2.1",
    description="API d'analyse de données cardiaques avec algorithme Apriori optimisé"
)

# Configuration CORS améliorée
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
    min_support: Optional[float] = Field(
        default=None, 
        ge=0.01, 
        le=1.0,
        description="Support minimal (auto si None)"
    )
    min_confidence: float = Field(
        default=0.7, 
        ge=0.0, 
        le=1.0,
        description="Confiance minimale pour les règles"
    )

class AprioriResponse(BaseModel):
    success: bool
    rules: List[Dict[str, Any]]
    attributes: List[str]
    statistics: Dict[str, Any]
    total_rules: int
    total_transactions: int
    execution_time: float
    matrix_type: str

class DatasetInfoResponse(BaseModel):
    total_transactions: int
    total_unique_items: int
    avg_transaction_length: float
    top_items: List[Dict[str, Any]]
    categories: Dict[str, int]

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


# =====================
# 🛠️ Utilitaires
# =====================

def to_native(obj):
    """
    Conversion robuste de tous types NumPy/SciPy en types Python natifs.
    Gère np.int64, np.float64, np.bool_, matrices creuses, etc.
    """
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
    🔬 Exécute l'algorithme Apriori optimisé avec:
    - Support adaptatif intelligent
    - Détection d'outliers statistiques
    - Matrices creuses/denses optimisées
    - Métriques avancées (lift, conviction, leverage)
    """
    try:
        logger.info(f"Démarrage Apriori avec params: {params}")
        
        # Chargement des données
        data_loader = HeartFailureDataLoader()
        transactions = data_loader.load_dataset()
        
        if not transactions:
            raise HTTPException(status_code=400, detail="Aucune transaction chargée")

        # Validation des paramètres
        if params.min_support is not None and not (0.01 <= params.min_support <= 1.0):
            raise HTTPException(
                status_code=400, 
                detail="min_support doit être entre 0.01 et 1.0"
            )

        # Exécution de l'algorithme
        apriori = Apriori(
            min_support=params.min_support,
            min_confidence=params.min_confidence
        )
        apriori.fit(transactions)
        rules = apriori.generate_rules()
        statistics = apriori.get_statistics()

        # Conversion en types Python natifs
        rules_native = to_native(rules)
        statistics_native = to_native(statistics)
        
        # Extraction des attributs uniques
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
        
        logger.info(f"Apriori terminé: {len(rules_native)} règles, {statistics_native['execution_time']:.2f}s")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution d'Apriori: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur interne: {str(e)}"
        )


@app.get("/dataset_info", response_model=DatasetInfoResponse)
async def get_dataset_info():
    """
    📊 Informations détaillées sur le dataset:
    - Nombre de transactions (patients)
    - Items uniques et fréquences
    - Distribution par catégories
    - Top items les plus fréquents
    """
    try:
        logger.info("Récupération des informations du dataset")
        
        data_loader = HeartFailureDataLoader()
        transactions = data_loader.load_dataset()
        
        if not transactions:
            raise HTTPException(status_code=400, detail="Dataset vide ou non trouvé")
        
        # Analyse des items
        all_items = [item for t in transactions for item in t]
        item_counts = Counter(all_items)
        
        # Regroupement par catégories
        categories: Dict[str, int] = defaultdict(int)
        for item in item_counts.keys():
            category = item.split('_')[0] if '_' in item else "other"
            categories[category] += 1
        
        # Statistiques
        transaction_lengths = [len(t) for t in transactions]
        avg_len = np.mean(transaction_lengths)
        
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
            categories={k: int(v) for k, v in dict(categories).items()}
        )
        
        logger.info(f"Dataset info: {len(transactions)} transactions, {len(item_counts)} items")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des infos dataset: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    🩺 Vérification de la santé du service
    """
    return HealthResponse(
        status="healthy",
        service="CardioAI API",
        version="2.1"
    )


@app.get("/")
async def root():
    """
    📌 Point d'entrée principal avec documentation
    """
    return {
        "service": "CardioAI API",
        "version": "2.1",
        "description": "API d'analyse de données cardiaques avec Apriori",
        "endpoints": {
            "POST /run_apriori": "Exécuter l'algorithme Apriori",
            "GET /dataset_info": "Informations sur le dataset",
            "GET /health": "Vérification de santé",
            "GET /docs": "Documentation interactive Swagger"
        },
        "features": [
            "Support adaptatif intelligent",
            "Détection d'outliers statistiques",
            "Matrices optimisées (sparse/dense)",
            "Métriques avancées (lift, conviction, leverage)"
        ]
    }


@app.on_event("startup")
async def startup_event():
    """
    🚀 Initialisation au démarrage
    """
    logger.info("=" * 70)
    logger.info("🚀 CardioAI API v2.1 démarrage...")
    logger.info("=" * 70)
    
    try:
        # Vérification du dataset
        data_loader = HeartFailureDataLoader()
        transactions = data_loader.load_dataset()
        logger.info(f"✅ Dataset chargé: {len(transactions)} transactions")
    except Exception as e:
        logger.warning(f"⚠️  Impossible de charger le dataset au démarrage: {e}")
    
    logger.info("=" * 70)


@app.on_event("shutdown")
async def shutdown_event():
    """
    🛑 Nettoyage à l'arrêt
    """
    logger.info("🛑 Arrêt de CardioAI API v2.1")


# =====================
# ⚙️ Point d'entrée
# =====================
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 70)
    print("🚀 Démarrage du serveur CardioAI API v2.1")
    print("=" * 70)
    print("📡 URL: http://127.0.0.1:8000")
    print("📚 Documentation: http://127.0.0.1:8000/docs")
    print("🔬 Endpoints:")
    print("   • POST /run_apriori")
    print("   • GET  /dataset_info")
    print("   • GET  /health")
    print("=" * 70 + "\n")
    
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )