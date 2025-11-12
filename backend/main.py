from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from core.apriori import Apriori
from data.loader import HeartFailureDataLoader

app = FastAPI()

# ========================
# 🔧 Configuration CORS
# ========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# 📦 Modèle de paramètres
# ========================
class AprioriParams(BaseModel):
    min_support: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)

# ========================
# 🚀 Endpoint principal
# ========================
@app.post("/run_apriori")
async def run_apriori(params: AprioriParams):
    """
    Exécute l'algorithme Apriori sur le dataset de maladies cardiaques.
    """
    try:
        # Chargement des données
        data_loader = HeartFailureDataLoader()
        transactions = data_loader.load_dataset()
        
        if not transactions:
            raise HTTPException(status_code=400, detail="Aucune transaction chargée")

        # Initialisation d'Apriori (min_support peut être None)
        apriori = Apriori(
            min_confidence=params.min_confidence,
            min_support=params.min_support  # None → support adaptatif
        )

        # Exécution de l'algorithme
        apriori.fit(transactions)
        rules = apriori.generate_rules()

        # Extraction des attributs distincts
        attributes = set()
        for rule in rules:
            attributes.update(rule['antecedent'])
            attributes.update(rule['consequent'])
        
        return {
            "success": True,
            "rules": rules,
            "attributes": sorted(list(attributes)),
            "total_rules": len(rules),
            "total_transactions": len(transactions)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'exécution : {str(e)}")

# ========================
# ❤️ Endpoint de santé
# ========================
@app.get("/health")
async def health_check():
    """Vérification de l'état du service"""
    return {"status": "healthy", "service": "CardioAI API"}

# ========================
# ▶️ Exécution locale
# ========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
