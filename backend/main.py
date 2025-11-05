from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.apriori import Apriori  # selon ton import
from data.loader import HeartFailureDataLoader  # selon ton import

app = FastAPI()

# Autoriser le frontend React
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # ou ["*"] pour tout autoriser (moins sécurisé)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/run_apriori")
async def run_apriori(params: dict):
    min_support = params.get("min_support", 0.05)
    min_confidence = params.get("min_confidence", 0.7)

    # Charge le dataset et exécute Apriori
    data_loader = HeartFailureDataLoader()
    transactions = data_loader.load_dataset()

    apriori = Apriori(min_confidence=min_confidence, min_support=min_support)
    apriori.fit(transactions)
    rules = apriori.generate_rules()
    
    # Retourne les règles pour le frontend
    return {"rules": rules}

# Lancer le serveur avec Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000)

