from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
from scipy.spatial.distance import cdist
import random

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CardioAI API",
    version="2.3",
    description="API d'analyse de données cardiaques avec clustering interactif"
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

class InteractiveClusteringParams(BaseModel):
    selected_points: List[int]
    min_distance: float
    auto_distance: bool = False

class KMeansInitParams(BaseModel):
    n_clusters: int
    initial_centroids: List[List[float]]

# =====================
# 🗄️ Gestion des données
# =====================

class HeartDataManager:
    def __init__(self):
        self.data = None
        self.scaled_data = None
        self.scaler = None
        self.pca_2d = None
        self.pca_3d = None
        self.data_2d = None
        self.data_3d = None
        self.feature_names = None
        self.load_data()
    
    def load_data(self):
        """Charge et prépare les données"""
        try:
            # Charger depuis le CSV
            df = pd.read_csv('heart_failure.csv')  # Assurez-vous que le fichier est dans le même dossier
            
            # Extraire les colonnes numériques
            numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
            
            # Vérifier que les colonnes existent
            available_numeric = [col for col in numeric_cols if col in df.columns]
            
            if not available_numeric:
                raise ValueError("Aucune colonne numérique trouvée")
            
            self.data = df[available_numeric].copy()
            self.feature_names = available_numeric
            
            # Standardisation
            self.scaler = StandardScaler()
            self.scaled_data = self.scaler.fit_transform(self.data)
            
            # Réduction de dimension pour la visualisation
            self.pca_2d = PCA(n_components=2)
            self.data_2d = self.pca_2d.fit_transform(self.scaled_data)
            
            self.pca_3d = PCA(n_components=3)
            self.data_3d = self.pca_3d.fit_transform(self.scaled_data)
            
            logger.info(f"Données chargées: {len(self.data)} échantillons, {len(self.feature_names)} features")
            
        except Exception as e:
            logger.error(f"Erreur chargement données: {e}")
            # Créer des données de test si le fichier n'existe pas
            self._create_sample_data()
    
    def _create_sample_data(self):
        """Crée des données de test pour le développement"""
        n_samples = 100
        n_features = 6
        
        np.random.seed(42)
        self.data = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'Feature_{i}' for i in range(n_features)]
        )
        self.feature_names = list(self.data.columns)
        
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(self.data)
        
        self.pca_2d = PCA(n_components=2)
        self.data_2d = self.pca_2d.fit_transform(self.scaled_data)
        
        logger.info(f"Données de test créées: {len(self.data)} échantillons")
    
    def get_scaled_point(self, index: int) -> List[float]:
        """Retourne un point standardisé"""
        if index < 0 or index >= len(self.scaled_data):
            raise ValueError(f"Index {index} hors limites")
        return self.scaled_data[index].tolist()
    
    def get_2d_point(self, index: int) -> List[float]:
        """Retourne un point en 2D pour visualisation"""
        if index < 0 or index >= len(self.data_2d):
            raise ValueError(f"Index {index} hors limites")
        return self.data_2d[index].tolist()

# Instance globale du gestionnaire de données
data_manager = HeartDataManager()

# =====================
# 🔬 Algorithmes de clustering
# =====================

class DistanceBasedClustering:
    """Clustering basé sur la distance avec sélection interactive"""
    
    @staticmethod
    def calculate_auto_distance(data: np.ndarray) -> float:
        """
        Calcule automatiquement la distance minimale optimale
        Basé sur l'article: A Distributed Clustering with Intelligent Multi Agents System
        """
        n_samples = len(data)
        
        if n_samples < 10:
            return 0.5
        
        # Échantillonner aléatoirement pour accélérer le calcul
        sample_size = min(100, n_samples)
        indices = random.sample(range(n_samples), sample_size)
        sample_data = data[indices]
        
        # Calculer les distances aux k plus proches voisins
        k = min(5, sample_size - 1)
        distances = []
        
        for i in range(sample_size):
            # Calculer les distances à tous les autres points
            point_distances = np.linalg.norm(sample_data - sample_data[i], axis=1)
            # Trier et prendre les k plus proches (exclure le point lui-même)
            sorted_distances = np.sort(point_distances)[1:k+1]
            distances.extend(sorted_distances.tolist())
        
        if not distances:
            return 1.0
        
        # Utiliser la médiane pour être robuste aux outliers
        auto_distance = float(np.median(distances))
        
        # Ajuster basé sur la densité
        avg_distance = float(np.mean(distances))
        density_factor = avg_distance / auto_distance
        
        # Formule basée sur l'article (adaptée)
        optimal_distance = auto_distance * (1 + 0.1 * np.log1p(density_factor))
        
        return round(optimal_distance, 3)
    
    @staticmethod
    def cluster_by_distance(data: np.ndarray, seed_indices: List[int], 
                          min_distance: float, feature_names: List[str]) -> Dict[str, Any]:
        """
        Effectue le clustering basé sur la distance
        
        Args:
            data: Données standardisées
            seed_indices: Indices des points sélectionnés comme graines
            min_distance: Distance minimale pour regrouper
            feature_names: Noms des features
        
        Returns:
            Dictionnaire avec les clusters et statistiques
        """
        n_samples = len(data)
        
        if not seed_indices:
            raise ValueError("Au moins un point de graine est requis")
        
        # Initialiser les clusters avec les graines
        clusters = []
        labels = -np.ones(n_samples, dtype=int)
        
        # Chaque graine forme son propre cluster initial
        for i, seed_idx in enumerate(seed_indices):
            if labels[seed_idx] == -1:  # Pas encore assigné
                clusters.append({
                    'id': i,
                    'seed_index': seed_idx,
                    'indices': [seed_idx],
                    'centroid': data[seed_idx].tolist()
                })
                labels[seed_idx] = i
        
        # Assigner les autres points
        for point_idx in range(n_samples):
            if labels[point_idx] != -1:
                continue  # Déjà assigné
            
            point = data[point_idx]
            min_dist = float('inf')
            closest_cluster = -1
            
            # Trouver le cluster le plus proche
            for cluster in clusters:
                centroid = np.array(cluster['centroid'])
                dist = np.linalg.norm(point - centroid)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_cluster = cluster['id']
            
            # Vérifier la distance minimale
            if min_dist <= min_distance:
                # Assigner au cluster
                labels[point_idx] = closest_cluster
                clusters[closest_cluster]['indices'].append(point_idx)
                
                # Mettre à jour le centroïde
                cluster_indices = clusters[closest_cluster]['indices']
                new_centroid = np.mean(data[cluster_indices], axis=0)
                clusters[closest_cluster]['centroid'] = new_centroid.tolist()
            else:
                # Créer un nouveau cluster avec ce point
                new_id = len(clusters)
                clusters.append({
                    'id': new_id,
                    'seed_index': point_idx,
                    'indices': [point_idx],
                    'centroid': point.tolist()
                })
                labels[point_idx] = new_id
        
        # Calculer les statistiques
        stats = {
            'n_clusters': len(clusters),
            'n_samples': n_samples,
            'min_distance': min_distance,
            'seed_points': seed_indices
        }
        
        # Enrichir les informations des clusters
        for cluster in clusters:
            indices = cluster['indices']
            cluster_data = data[indices]
            
            cluster['size'] = len(indices)
            cluster['percentage'] = (len(indices) / n_samples) * 100
            
            # Calculer les stats par feature
            cluster['features'] = {}
            for i, feature in enumerate(feature_names):
                values = cluster_data[:, i]
                cluster['features'][feature] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
            
            # Points dans l'espace 2D
            cluster['points_2d'] = data_manager.data_2d[indices].tolist()
        
        return {
            'clusters': clusters,
            'labels': labels.tolist(),
            'statistics': stats
        }

# =====================
# 🚀 Endpoints API
# =====================

@app.get("/dataset_info")
async def get_dataset_info():
    """📊 Informations sur le dataset"""
    try:
        return {
            'success': True,
            'total_samples': len(data_manager.data),
            'features': data_manager.feature_names,
            'feature_stats': {
                feature: {
                    'mean': float(data_manager.data[feature].mean()),
                    'std': float(data_manager.data[feature].std()),
                    'min': float(data_manager.data[feature].min()),
                    'max': float(data_manager.data[feature].max())
                }
                for feature in data_manager.feature_names
            }
        }
    except Exception as e:
        logger.error(f"Erreur dataset info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_data_points")
async def get_data_points():
    """📈 Récupère les points de données pour visualisation"""
    try:
        points_2d = data_manager.data_2d.tolist()
        points_3d = data_manager.data_3d.tolist()
        scaled_data = data_manager.scaled_data.tolist()
        
        return {
            'success': True,
            'points_2d': points_2d,
            'points_3d': points_3d,
            'scaled_data': scaled_data,
            'feature_names': data_manager.feature_names,
            'n_samples': len(points_2d)
        }
    except Exception as e:
        logger.error(f"Erreur get_data_points: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/calculate_auto_distance")
async def calculate_auto_distance():
    """🤖 Calcule la distance minimale automatique"""
    try:
        auto_distance = DistanceBasedClustering.calculate_auto_distance(
            data_manager.scaled_data
        )
        
        return {
            'success': True,
            'auto_distance': auto_distance,
            'message': f"Distance automatique calculée: {auto_distance}"
        }
    except Exception as e:
        logger.error(f"Erreur calculate_auto_distance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/interactive_clustering")
async def interactive_clustering(params: InteractiveClusteringParams):
    """🎯 Clustering interactif basé sur la distance"""
    try:
        logger.info(f"Clustering interactif: {params.selected_points} points, distance={params.min_distance}")
        
        # Utiliser la distance automatique si demandé
        if params.auto_distance:
            auto_dist = DistanceBasedClustering.calculate_auto_distance(
                data_manager.scaled_data
            )
            min_distance = auto_dist
        else:
            min_distance = params.min_distance
        
        # Effectuer le clustering
        result = DistanceBasedClustering.cluster_by_distance(
            data=data_manager.scaled_data,
            seed_indices=params.selected_points,
            min_distance=min_distance,
            feature_names=data_manager.feature_names
        )
        
        # Ajouter les points 2D pour chaque point
        result['points_2d'] = data_manager.data_2d.tolist()
        
        return {
            'success': True,
            'clustering_result': result,
            'used_distance': min_distance,
            'auto_calculated': params.auto_distance
        }
    except Exception as e:
        logger.error(f"Erreur interactive_clustering: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/initialize_kmeans_from_clusters")
async def initialize_kmeans_from_clusters(params: KMeansInitParams):
    """🚀 Initialise K-Means avec des clusters existants"""
    try:
        from core.kmeans import KMeans
        
        # Vérifier les centroïdes initiaux
        if not params.initial_centroids:
            raise ValueError("Centroïdes initiaux requis")
        
        # Vérifier la cohérence des dimensions
        n_features = len(data_manager.scaled_data[0])
        for centroid in params.initial_centroids:
            if len(centroid) != n_features:
                raise ValueError(f"Centroïde de dimension {len(centroid)}, attendu {n_features}")
        
        # Exécuter K-Means avec les centroïdes initiaux
        kmeans = KMeans(
            n_clusters=params.n_clusters,
            max_iterations=100,
            random_state=42
        )
        
        # Utiliser une méthode modifiée pour accepter des centroïdes initiaux
        # (Vous devrez modifier votre classe KMeans pour accepter ce paramètre)
        kmeans.centroids = np.array(params.initial_centroids)
        kmeans.fit(data_manager.scaled_data)
        
        # Calculer le score de silhouette
        silhouette = kmeans.calculate_silhouette_score(data_manager.scaled_data)
        statistics = kmeans.get_statistics()
        
        # Formater les résultats
        clusters_info = []
        for i in range(params.n_clusters):
            cluster_mask = kmeans.labels == i
            cluster_data = data_manager.scaled_data[cluster_mask]
            
            clusters_info.append({
                'cluster_id': i,
                'size': int(np.sum(cluster_mask)),
                'percentage': float(np.sum(cluster_mask) / len(data_manager.scaled_data) * 100),
                'centroid': kmeans.centroids[i].tolist(),
                'features': {
                    feature: {
                        'mean': float(data_manager.data[feature].iloc[cluster_mask].mean()),
                        'std': float(data_manager.data[feature].iloc[cluster_mask].std()),
                        'min': float(data_manager.data[feature].iloc[cluster_mask].min()),
                        'max': float(data_manager.data[feature].iloc[cluster_mask].max())
                    }
                    for feature in data_manager.feature_names
                }
            })
        
        return {
            'success': True,
            'clusters': clusters_info,
            'statistics': {
                'n_clusters': int(params.n_clusters),
                'n_iterations': int(statistics['n_iterations']),
                'execution_time': float(statistics['execution_time']),
                'inertia': float(statistics['inertia']),
                'silhouette_score': float(silhouette),
                'converged': bool(statistics['converged'])
            },
            'labels': kmeans.labels.tolist(),
            'centroids': kmeans.centroids.tolist(),
            'message': f"K-Means initialisé avec {params.n_clusters} clusters"
        }
        
    except Exception as e:
        logger.error(f"Erreur initialize_kmeans_from_clusters: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """🩺 Vérification de la santé du service"""
    return {
        'status': 'healthy',
        'service': 'CardioAI Interactive Clustering API',
        'version': '2.3',
        'dataset_loaded': data_manager.data is not None,
        'n_samples': len(data_manager.data) if data_manager.data else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)