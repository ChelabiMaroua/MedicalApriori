import numpy as np
from typing import List, Dict, Optional, Tuple
import time
from collections import defaultdict


class KMeans:
    """
    Implémentation optimisée de K-Means pour données médicales
    avec initialisation K-Means++ et métriques avancées
    """
    
    def __init__(self, n_clusters: int = 3, max_iterations: int = 100, 
                 tolerance: float = 1e-4, random_state: Optional[int] = None):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia = None
        self.n_iterations = 0
        self.execution_time = 0
        self.cluster_stats = {}
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _initialize_centroids_plus_plus(self, X: np.ndarray) -> np.ndarray:
        """
        Initialisation K-Means++ pour de meilleurs centroïdes initiaux.
        Sélectionne les centroïdes de manière à maximiser la distance entre eux.
        """
        n_samples = X.shape[0]
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        # Premier centroïde aléatoire
        centroids[0] = X[np.random.randint(n_samples)]
        
        # Sélection des k-1 centroïdes restants
        for i in range(1, self.n_clusters):
            # Calcul des distances au centroïde le plus proche
            distances = np.array([
                min([np.linalg.norm(x - c) ** 2 for c in centroids[:i]]) 
                for x in X
            ])
            
            # Probabilité proportionnelle au carré de la distance
            probabilities = distances / distances.sum()
            cumulative_probs = probabilities.cumsum()
            r = np.random.rand()
            
            # Sélection du prochain centroïde
            for j, p in enumerate(cumulative_probs):
                if r < p:
                    centroids[i] = X[j]
                    break
        
        return centroids
    
    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """
        Assigne chaque point au centroïde le plus proche.
        """
        distances = np.zeros((X.shape[0], self.n_clusters))
        
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Recalcule les centroïdes comme moyenne des points de chaque cluster.
        """
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                # Si un cluster est vide, réinitialiser avec un point aléatoire
                new_centroids[i] = X[np.random.randint(X.shape[0])]
        
        return new_centroids
    
    def _calculate_inertia(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Calcule l'inertie (somme des distances au carré aux centroïdes).
        """
        inertia = 0.0
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[i]) ** 2)
        return inertia
    
    def fit(self, X: np.ndarray) -> 'KMeans':
        """
        Entraîne le modèle K-Means sur les données X.
        
        Args:
            X: Matrice de données (n_samples, n_features)
            
        Returns:
            self: Instance entraînée
        """
        start_time = time.time()
        
        print(f"\n🎯 Démarrage K-Means")
        print("=" * 70)
        print(f"📊 Données: {X.shape[0]} échantillons, {X.shape[1]} features")
        print(f"🔢 Nombre de clusters: {self.n_clusters}")
        print(f"⚙️ Initialisation: K-Means++")
        
        # Initialisation K-Means++
        self.centroids = self._initialize_centroids_plus_plus(X)
        
        # Itérations
        for iteration in range(self.max_iterations):
            # Assignation des clusters
            new_labels = self._assign_clusters(X)
            
            # Mise à jour des centroïdes
            new_centroids = self._update_centroids(X, new_labels)
            
            # Vérification de la convergence
            centroid_shift = np.linalg.norm(new_centroids - self.centroids)
            
            if iteration % 10 == 0 or centroid_shift < self.tolerance:
                inertia = self._calculate_inertia(X, new_labels)
                print(f"📈 Itération {iteration}: Inertie = {inertia:.2f}, "
                      f"Déplacement = {centroid_shift:.6f}")
            
            self.centroids = new_centroids
            self.labels = new_labels
            self.n_iterations = iteration + 1
            
            # Convergence atteinte
            if centroid_shift < self.tolerance:
                print(f"✅ Convergence atteinte à l'itération {iteration}")
                break
        
        # Calcul de l'inertie finale
        self.inertia = self._calculate_inertia(X, self.labels)
        
        # Statistiques par cluster
        self._calculate_cluster_statistics(X)
        
        self.execution_time = time.time() - start_time
        
        print(f"\n✅ K-Means terminé en {self.execution_time:.2f}s")
        print(f"🎯 Inertie finale: {self.inertia:.2f}")
        print(f"🔄 Itérations: {self.n_iterations}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit le cluster le plus proche pour de nouveaux points.
        """
        if self.centroids is None:
            raise ValueError("Le modèle doit être entraîné avant la prédiction")
        
        return self._assign_clusters(X)
    
    def _calculate_cluster_statistics(self, X: np.ndarray):
        """
        Calcule des statistiques détaillées pour chaque cluster.
        """
        self.cluster_stats = {}
        
        for i in range(self.n_clusters):
            cluster_points = X[self.labels == i]
            
            if len(cluster_points) > 0:
                self.cluster_stats[i] = {
                    'size': int(len(cluster_points)),
                    'percentage': float(len(cluster_points) / len(X) * 100),
                    'centroid': self.centroids[i].tolist(),
                    'std': np.std(cluster_points, axis=0).tolist(),
                    'min': np.min(cluster_points, axis=0).tolist(),
                    'max': np.max(cluster_points, axis=0).tolist(),
                    'mean': np.mean(cluster_points, axis=0).tolist()
                }
            else:
                self.cluster_stats[i] = {
                    'size': 0,
                    'percentage': 0.0,
                    'centroid': self.centroids[i].tolist(),
                    'std': [0] * X.shape[1],
                    'min': [0] * X.shape[1],
                    'max': [0] * X.shape[1],
                    'mean': [0] * X.shape[1]
                }
    
    def get_statistics(self) -> Dict:
        """
        Retourne des statistiques détaillées du clustering.
        """
        return {
            'n_clusters': int(self.n_clusters),
            'n_iterations': int(self.n_iterations),
            'execution_time': float(self.execution_time),
            'inertia': float(self.inertia),
            'cluster_stats': self.cluster_stats,
            'converged': bool(self.n_iterations < self.max_iterations)
        }
    
    def calculate_silhouette_score(self, X: np.ndarray) -> float:
        """
        Calcule le score de silhouette pour évaluer la qualité du clustering.
        Score entre -1 et 1, plus c'est élevé, mieux c'est.
        """
        if self.labels is None:
            raise ValueError("Le modèle doit être entraîné avant le calcul du score")
        
        n_samples = X.shape[0]
        silhouette_scores = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Points du même cluster
            same_cluster = X[self.labels == self.labels[i]]
            
            # Distance moyenne intra-cluster (a)
            if len(same_cluster) > 1:
                a = np.mean([np.linalg.norm(X[i] - x) for x in same_cluster if not np.array_equal(X[i], x)])
            else:
                a = 0
            
            # Distance moyenne au cluster le plus proche (b)
            b = float('inf')
            for j in range(self.n_clusters):
                if j != self.labels[i]:
                    other_cluster = X[self.labels == j]
                    if len(other_cluster) > 0:
                        mean_dist = np.mean([np.linalg.norm(X[i] - x) for x in other_cluster])
                        b = min(b, mean_dist)
            
            # Score de silhouette
            if max(a, b) > 0:
                silhouette_scores[i] = (b - a) / max(a, b)
            else:
                silhouette_scores[i] = 0
        
        return float(np.mean(silhouette_scores))