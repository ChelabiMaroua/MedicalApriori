from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.apriori import Apriori
from data.loader import HeartFailureDataLoader

app = FastAPI()

# Autoriser toutes les origines (OK pour le développement)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # à restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/run_apriori")
@app.post("/run_apriori")
async def run_apriori(params: dict):
    min_support = params.get("min_support", 0.05)
    min_confidence = params.get("min_confidence", 0.7)

    data_loader = HeartFailureDataLoader()
    transactions = data_loader.load_dataset()

    apriori = Apriori(min_confidence=min_confidence, min_support=min_support)
    apriori.fit(transactions)
    rules = apriori.generate_rules()

    # Extraire les attributs distincts des règles
    attributes = set()
    for rule in rules:
        attributes.update(rule['antecedent'])
        attributes.update(rule['consequent'])
    
    return {
        "rules": rules,
        "attributes": list(attributes)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
