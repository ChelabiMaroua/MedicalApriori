from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from core.apriori import Apriori
from data.loader import HeartFailureDataLoader
from collections import Counter, defaultdict

app = FastAPI(title="CardioAI API", version="2.0")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔒 À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================
# 📦 Modèles Pydantic
# =====================

class AprioriParams(BaseModel):
    min_support: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)

class AprioriResponse(BaseModel):
    success: bool
    rules: List[Dict[str, Any]]
    attributes: List[str]
    statistics: Dict[str, Any]
    total_rules: int
    total_transactions: int


# =====================
# 🚀 Endpoints API
# =====================

@app.post("/run_apriori", response_model=AprioriResponse)
async def run_apriori(params: AprioriParams):
    """
    Exécute l'algorithme Apriori avec statistiques détaillées.
    Convertit tous les types NumPy et matrices creuses en types Python natifs
    pour éviter les erreurs de sérialisation Pydantic.
    """
    try:
        # Chargement des données
        data_loader = HeartFailureDataLoader()
        transactions = data_loader.load_dataset()
        
        if not transactions:
            raise HTTPException(status_code=400, detail="Aucune transaction chargée.")

        # Exécution de l’algorithme
        apriori = Apriori(
            min_support=params.min_support,
            min_confidence=params.min_confidence
        )
        apriori.fit(transactions)
        rules = apriori.generate_rules()
        statistics = apriori.get_statistics()

        # Fonction robuste de conversion en types Python natifs
        def to_native(obj):
            import numpy as np
            from scipy.sparse import spmatrix

            if isinstance(obj, np.generic):  # np.int64, np.float64, np.bool_, etc.
                return obj.item()
            elif isinstance(obj, (np.ndarray, list, tuple)):
                return [to_native(i) for i in obj]
            elif isinstance(obj, dict):
                return {str(k): to_native(v) for k, v in obj.items()}
            elif isinstance(obj, spmatrix):  # matrice creuse → liste de listes
                return obj.toarray().tolist()
            elif isinstance(obj, (bool, int, float, str)):
                return obj
            return str(obj)  # fallback pour tout autre type inconnu

        # Conversion
        rules_python = to_native(rules)
        statistics_python = to_native(statistics)
        attributes = sorted({str(item) for rule in rules_python for item in rule["antecedent"] + rule["consequent"]})

        return AprioriResponse(
            success=True,
            rules=rules_python,
            attributes=attributes,
            statistics=statistics_python,
            total_rules=len(rules_python),
            total_transactions=len(transactions)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur : {str(e)}")


@app.get("/dataset_info")
async def get_dataset_info():
    """📊 Renvoie des informations descriptives sur le dataset."""
    try:
        data_loader = HeartFailureDataLoader()
        transactions = data_loader.load_dataset()
        
        if not transactions:
            raise HTTPException(status_code=400, detail="Dataset vide ou non trouvé.")
        
        all_items = [item for t in transactions for item in t]
        item_counts = Counter(all_items)
        
        # Regroupement par catégories
        categories: Dict[str, int] = defaultdict(int)
        for item in item_counts.keys():
            category = item.split('_')[0] if '_' in item else item
            categories[category] += 1
        
        avg_len = sum(len(t) for t in transactions) / len(transactions)

        return {
            "total_transactions": len(transactions),
            "total_unique_items": len(item_counts),
            "avg_transaction_length": avg_len,
            "top_items": [
                {"item": item, "count": count, "percentage": round(count / len(transactions) * 100, 2)}
                for item, count in item_counts.most_common(10)
            ],
            "categories": dict(categories)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur : {str(e)}")


@app.get("/health")
async def health_check():
    """🩺 Vérifie la disponibilité du service."""
    return {"status": "healthy", "service": "CardioAI API v2.0"}


# =====================
# ⚙️ Point d'entrée
# =====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
