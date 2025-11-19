from collections import defaultdict
from itertools import combinations
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
import time
from typing import List, Dict, Set, FrozenSet, Tuple


class Apriori:
    def __init__(self, min_support=None, min_confidence=0.6):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.transactions = []
        self.frequent_itemsets = {}
        self.rules = []
        self.n_transactions = 0
        self.use_sparse = False
        self.transaction_matrix = None
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.execution_time = 0
        self.initial_support = None
        self.support_history = []
        self.iteration_stats = []
        
    def fit(self, transactions: List[List[str]]):
        """Exécuter l'algorithme Apriori avec adaptation automatique du support."""
        start_time = time.time()
        self.transactions = transactions
        self.n_transactions = len(transactions)

        if self.n_transactions == 0:
            raise ValueError("❌ Aucune transaction fournie")

        # Calcul automatique du support initial
        if self.min_support is None:
            self.calculate_adaptive_support()
        self.initial_support = self.min_support

        # Représentation matricielle (dense ou creuse)
        self._prepare_matrix_representation()

        print(f"\n🔍 Démarrage Apriori")
        print("=" * 70)
        print(f"📊 Matrice utilisée : {'CREUSE (optimisée pour sparse)' if self.use_sparse else 'DENSE (optimisée pour données denses)'}")
        print(f"🎯 Support initial : {self.initial_support:.4f}")
        print(f"📈 Stratégie : Support adaptatif avec filtrage statistique")

        # Itemsets de taille 1
        item_counts = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counts[frozenset([item])] += 1

        frequent_1 = {
            itemset: count / self.n_transactions
            for itemset, count in item_counts.items()
            if (count / self.n_transactions) >= self.min_support
        }

        if not frequent_1:
            print("⚠️ Aucun itemset fréquent trouvé. Essayez de réduire min_support.")
            return self

        self.frequent_itemsets[1] = [
            {'itemset': list(itemset), 'support': sup}
            for itemset, sup in frequent_1.items()
        ]

        self.iteration_stats.append({
            'iteration': 1,
            'candidates': len(item_counts),
            'frequent': len(frequent_1),
            'support': self.min_support,
            'pruning_rate': 1 - (len(frequent_1) / len(item_counts))
        })

        print(f"📊 Itération 1 : {len(frequent_1)} itemsets fréquents / {len(item_counts)} candidats "
              f"(taux de filtrage: {(1 - len(frequent_1)/len(item_counts))*100:.1f}%)")

        # Étapes suivantes
        k = 2
        current_frequent = frequent_1

        while current_frequent:
            candidates = self._generate_candidates(current_frequent, k)
            if not candidates:
                print(f"⚠️ Aucun candidat généré pour k={k}. Arrêt.")
                break

            # Comptage des supports (optimisé selon le type de matrice)
            if self.use_sparse:
                candidate_counts = self._count_support_sparse(candidates)
            else:
                candidate_counts = self._count_support_dense(candidates)

            if not candidate_counts:
                print(f"⚠️ Aucun candidat avec support suffisant pour k={k}. Arrêt.")
                break

            # Recalcul dynamique et intelligent du support
            all_supports = np.array([c / self.n_transactions for c in candidate_counts.values()])
            mean_support = np.mean(all_supports)
            median_support = np.median(all_supports)
            std_support = np.std(all_supports)
            
            # Filtrage des outliers (valeurs extrêmes)
            # On utilise l'écart interquartile pour détecter les valeurs aberrantes
            q1 = np.percentile(all_supports, 25)
            q3 = np.percentile(all_supports, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Filtrer les valeurs dans l'intervalle interquartile
            filtered_supports = all_supports[(all_supports >= lower_bound) & (all_supports <= upper_bound)]
            
            if len(filtered_supports) > 0:
                robust_mean = np.mean(filtered_supports)
                robust_std = np.std(filtered_supports)
                outliers_removed = len(all_supports) - len(filtered_supports)
            else:
                robust_mean = mean_support
                robust_std = std_support
                outliers_removed = 0

            # Calcul du nouveau support avec pondération adaptative
            # On donne plus de poids à la moyenne robuste qu'à l'ancienne valeur
            prev_support = self.min_support
            alpha = 0.3  # Poids de l'ancien support (inertie)
            beta = 0.7   # Poids du nouveau calcul
            
            new_support = alpha * prev_support + beta * robust_mean
            
            # Contraintes : entre 1% et 50%, et ne pas augmenter trop vite
            new_support = max(0.01, min(0.5, new_support))
            
            # Éviter les sauts trop brusques
            max_increase = 1.5  # Maximum 50% d'augmentation
            max_decrease = 0.5  # Maximum 50% de diminution
            if new_support > prev_support * max_increase:
                new_support = prev_support * max_increase
            elif new_support < prev_support * max_decrease:
                new_support = prev_support * max_decrease
                
            self.min_support = new_support

            self.support_history.append({
                'iteration': k,
                'support': self.min_support,
                'mean': float(mean_support),
                'median': float(median_support),
                'robust_mean': float(robust_mean),
                'std': float(std_support),
                'robust_std': float(robust_std),
                'outliers_removed': int(outliers_removed),
                'q1': float(q1),
                'q3': float(q3)
            })

            print(f"\n⚙️  Itération {k} - Analyse statistique des supports:")
            print(f"   📊 Moyenne brute: {mean_support:.4f}, Médiane: {median_support:.4f}, σ: {std_support:.4f}")
            print(f"   🎯 Moyenne robuste: {robust_mean:.4f}, σ robuste: {robust_std:.4f}")
            print(f"   🔍 Outliers filtrés: {outliers_removed} ({outliers_removed/len(all_supports)*100:.1f}%)")
            print(f"   ✅ Nouveau support: {self.min_support:.4f} (variation: {((new_support/prev_support - 1)*100):+.1f}%)")

            # Sélection des nouveaux itemsets fréquents
            new_frequent = {
                itemset: count / self.n_transactions
                for itemset, count in candidate_counts.items()
                if (count / self.n_transactions) >= self.min_support
            }

            if not new_frequent:
                print(f"⚠️ Aucun itemset fréquent avec le nouveau support. Arrêt.")
                break

            self.frequent_itemsets[k] = [
                {'itemset': list(itemset), 'support': sup}
                for itemset, sup in new_frequent.items()
            ]

            pruning_rate = 1 - (len(new_frequent) / len(candidates))
            self.iteration_stats.append({
                'iteration': k,
                'candidates': len(candidates),
                'frequent': len(new_frequent),
                'support': self.min_support,
                'pruning_rate': pruning_rate
            })

            print(f"📊 Résultat : {len(new_frequent)} itemsets fréquents / {len(candidates)} candidats "
                  f"(taux de filtrage: {pruning_rate*100:.1f}%)")
            
            current_frequent = new_frequent
            k += 1

        self.execution_time = time.time() - start_time
        total_itemsets = sum(len(v) for v in self.frequent_itemsets.values())
        print(f"\n✅ Apriori terminé en {self.execution_time:.2f}s")
        print(f"📦 Total : {total_itemsets} itemsets fréquents sur {k-1} itérations")
        print(f"🎯 Support final : {self.min_support:.4f} (variation totale: {((self.min_support/self.initial_support - 1)*100):+.1f}%)")
        return self

    def _prepare_matrix_representation(self):
        """Prépare une représentation matricielle optimale (dense ou creuse)."""
        all_items = sorted({item for t in self.transactions for item in t})
        self.item_to_idx = {item: i for i, item in enumerate(all_items)}
        self.idx_to_item = {i: item for item, i in self.item_to_idx.items()}

        n_items = len(all_items)
        avg_len = np.mean([len(t) for t in self.transactions])
        density = avg_len / n_items if n_items > 0 else 0

        # Seuil de densité : < 30% → matrice creuse, sinon dense
        self.use_sparse = density < 0.3
        
        print(f"📊 Analyse de la matrice:")
        print(f"   • Items: {n_items}")
        print(f"   • Transactions: {self.n_transactions}")
        print(f"   • Taille moyenne: {avg_len:.2f} items/transaction")
        print(f"   • Densité: {density*100:.1f}%")
        print(f"   • Décision: Matrice {'CREUSE (CSR)' if self.use_sparse else 'DENSE (NumPy)'}")

        if self.use_sparse:
            # Construction matrice creuse (format CSR pour accès ligne rapide)
            mat = lil_matrix((self.n_transactions, n_items), dtype=bool)
            for t_idx, transaction in enumerate(self.transactions):
                for item in transaction:
                    mat[t_idx, self.item_to_idx[item]] = True
            self.transaction_matrix = mat.tocsr()
            
            print(f"   • Mémoire: ~{self.transaction_matrix.data.nbytes / 1024:.1f} KB (sparse)")
        else:
            # Construction matrice dense
            mat = np.zeros((self.n_transactions, n_items), dtype=bool)
            for t_idx, transaction in enumerate(self.transactions):
                for item in transaction:
                    mat[t_idx, self.item_to_idx[item]] = True
            self.transaction_matrix = mat
            
            print(f"   • Mémoire: ~{mat.nbytes / 1024:.1f} KB (dense)")

    def _count_support_sparse(self, candidates: Set[FrozenSet]) -> Dict[FrozenSet, int]:
        """Comptage optimisé du support avec matrice creuse (CSR)."""
        candidate_counts = defaultdict(int)
        X = self.transaction_matrix

        for candidate in candidates:
            indices = [self.item_to_idx[item] for item in candidate]
            
            if len(indices) == 1:
                # Cas simple : somme directe de la colonne
                count = X[:, indices[0]].sum()
            elif len(indices) == 2:
                # Optimisation pour paires : intersection via multiplication
                col1 = X[:, indices[0]].toarray().ravel()
                col2 = X[:, indices[1]].toarray().ravel()
                count = np.sum(col1 & col2)
            else:
                # Cas général : vérification AND de toutes les colonnes
                subset = X[:, indices].toarray()
                count = np.sum(np.all(subset, axis=1))
            
            candidate_counts[candidate] = int(count)

        return candidate_counts

    def _count_support_dense(self, candidates: Set[FrozenSet]) -> Dict[FrozenSet, int]:
        """Comptage optimisé du support avec matrice dense."""
        candidate_counts = defaultdict(int)
        X = self.transaction_matrix

        for candidate in candidates:
            indices = [self.item_to_idx[item] for item in candidate]
            
            if len(indices) == 1:
                count = np.sum(X[:, indices[0]])
            elif len(indices) == 2:
                # Optimisation vectorisée pour paires
                count = np.sum(X[:, indices[0]] & X[:, indices[1]])
            else:
                # Opération AND vectorisée sur toutes les colonnes
                count = np.sum(np.all(X[:, indices], axis=1))
            
            candidate_counts[candidate] = int(count)

        return candidate_counts

    def calculate_adaptive_support(self):
        """Calcul initial adaptatif et intelligent du support."""
        print("\n⚙️ Calcul du support minimal adaptatif initial")
        print("=" * 70)

        if not self.transactions:
            self.min_support = 0.05
            return self.min_support

        avg_len = np.mean([len(t) for t in self.transactions])
        unique_items = len({item for t in self.transactions for item in t})
        density = avg_len / unique_items if unique_items else 0

        # Stratégie adaptative selon la taille du dataset
        if self.n_transactions < 100:
            base = 0.15  # Petit dataset : support plus élevé
        elif self.n_transactions < 500:
            base = 0.08  # Dataset moyen
        else:
            base = 0.03  # Grand dataset : support plus bas

        # Ajustement selon la densité
        density_factor = max(0.01, min(0.1, density * 0.5))
        
        support = max(base, min(0.3, base + density_factor))

        self.min_support = support
        print(f"📊 Analyse:")
        print(f"   • Transactions: {self.n_transactions}")
        print(f"   • Items uniques: {unique_items}")
        print(f"   • Taille moyenne: {avg_len:.2f}")
        print(f"   • Densité: {density:.2%}")
        print(f"✅ Support initial calculé : {support:.4f}")
        return support

    def _generate_candidates(self, frequent_itemsets: Dict, k: int) -> Set[FrozenSet]:
        """Génère les candidats de taille k avec élagage Apriori."""
        candidates = set()
        itemsets = sorted([tuple(sorted(i)) for i in frequent_itemsets.keys()])

        for i in range(len(itemsets)):
            for j in range(i + 1, len(itemsets)):
                a, b = itemsets[i], itemsets[j]
                # Jointure uniquement si les k-2 premiers éléments sont identiques
                if a[:-1] == b[:-1]:
                    new = frozenset(a + (b[-1],))
                    # Élagage : vérifier que tous les sous-ensembles de taille k-1 sont fréquents
                    if self._has_frequent_subsets(new, frequent_itemsets):
                        candidates.add(new)
        return candidates

    def _has_frequent_subsets(self, candidate: FrozenSet, frequent_itemsets: Dict) -> bool:
        """Vérifie la propriété antimonotone (élagage Apriori)."""
        for item in candidate:
            subset = candidate - frozenset([item])
            if subset not in frequent_itemsets:
                return False
        return True

    def _get_support(self, itemset: FrozenSet) -> float:
        """Retourne le support d'un itemset."""
        k = len(itemset)
        if k in self.frequent_itemsets:
            for d in self.frequent_itemsets[k]:
                if frozenset(d['itemset']) == itemset:
                    return d['support']
        return 0.0

    def generate_rules(self) -> List[Dict]:
        """Génère les règles d'association avec métriques avancées."""
        print("\n🎯 Génération des règles d'association")
        print("=" * 70)
        self.rules = []

        for k, itemsets in self.frequent_itemsets.items():
            if k < 2:
                continue
            for data in itemsets:
                itemset = frozenset(data['itemset'])
                s_itemset = data['support']

                # Générer toutes les règles possibles
                for i in range(1, len(itemset)):
                    for antecedent in combinations(itemset, i):
                        antecedent = frozenset(antecedent)
                        consequent = itemset - antecedent
                        
                        s_ant = self._get_support(antecedent)
                        s_con = self._get_support(consequent)
                        
                        if s_ant == 0:
                            continue

                        confidence = s_itemset / s_ant
                        lift = confidence / s_con if s_con > 0 else 0
                        
                        # Conviction : mesure de l'implication
                        conviction = (1 - s_con) / (1 - confidence) if confidence < 1 else float('inf')
                        
                        # Leverage : différence entre support observé et attendu
                        leverage = s_itemset - (s_ant * s_con)

                        if confidence >= self.min_confidence:
                            self.rules.append({
                                'antecedent': list(antecedent),
                                'consequent': list(consequent),
                                'support': float(s_itemset),
                                'confidence': float(confidence),
                                'lift': float(lift),
                                'conviction': float(conviction) if conviction != float('inf') else 999.0,
                                'leverage': float(leverage)
                            })

        # Tri par confiance puis lift
        self.rules.sort(key=lambda x: (-x['confidence'], -x['lift']))
        
        print(f"✅ {len(self.rules)} règles générées (confiance ≥ {self.min_confidence})")
        if self.rules:
            print(f"📊 Top règle : {self.rules[0]['antecedent']} → {self.rules[0]['consequent']}")
            print(f"   Confiance: {self.rules[0]['confidence']:.2%}, Lift: {self.rules[0]['lift']:.2f}")
        return self.rules

    def get_statistics(self) -> Dict:
        """Retourne des statistiques détaillées d'exécution."""
        return {
            'execution_time': float(self.execution_time),
            'total_transactions': int(self.n_transactions),
            'total_rules': int(len(self.rules)),
            'total_frequent_itemsets': int(sum(len(v) for v in self.frequent_itemsets.values())),
            'initial_support': float(self.initial_support) if self.initial_support else None,
            'final_support': float(self.min_support),
            'support_variation_pct': float(((self.min_support / self.initial_support - 1) * 100)) if self.initial_support else 0,
            'use_sparse_matrix': bool(self.use_sparse),
            'support_history': self.support_history,
            'iteration_stats': self.iteration_stats,
            'itemsets_by_size': {int(k): int(len(v)) for k, v in self.frequent_itemsets.items()},
            'avg_pruning_rate': float(np.mean([s['pruning_rate'] for s in self.iteration_stats])) if self.iteration_stats else 0
        }