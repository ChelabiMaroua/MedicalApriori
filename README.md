# CardioAI - Medical Apriori & Clustering API

## Présentation

Ce projet propose une API FastAPI pour l'analyse de données médicales cardiaques, intégrant des algorithmes de clustering interactif et d'extraction d'associations (Apriori). Il est conçu pour être utilisé avec une interface frontend (non fournie ici) pour l'exploration et la visualisation des clusters.

- **Clustering interactif** : Sélection de points, clustering par distance, initialisation K-Means++.
- **Apriori** : Extraction de règles d'association sur des transactions médicales.
- **Dataset** : Données cardiaques (exemple : `heart.csv`).

## Structure du projet

```
backend/
  main.py                # API FastAPI (clustering, endpoints)
  core/
    apriori.py           # Algorithme Apriori
    kmeans.py            # Algorithme K-Means optimisé
  data/
    loader.py            # Chargement et transformation des données
    datasets/
      heart.csv          # Exemple de dataset cardiaque
  test_api.py            # Script de test des endpoints
```

## Installation

1. **Prérequis** : Python 3.8+, pip
2. Placez-vous dans le dossier `backend/` :

```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
# ou
source venv/bin/activate  # Linux/Mac
pip install fastapi uvicorn scikit-learn pandas numpy scipy pydantic
```

## Lancement de l'API

```bash
uvicorn main:app --reload
```
L'API sera accessible sur [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Endpoints principaux

- `GET /dataset_info` : Infos sur le dataset
- `GET /get_data_points` : Points pour visualisation
- `POST /calculate_auto_distance` : Calcul de la distance optimale
- `POST /interactive_clustering` : Clustering interactif
- `POST /initialize_kmeans_from_clusters` : K-Means à partir de clusters
- `GET /health` : Vérification de santé

## Tests

Utilisez `test_api.py` pour tester les endpoints :
```bash
python test_api.py
```

## Données

- Placez votre fichier CSV dans `backend/data/datasets/` (ex: `heart.csv`).
- Modifiez le chemin dans `loader.py` si besoin.

## Auteurs
- Projet CardioAI, 2026
