from collections import defaultdict
from itertools import combinations
import numpy as np
from scipy.sparse import lil_matrix
import time


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

    def fit(self, transactions):
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
        print(f"📊 Matrice utilisée : {'CREUSE' if self.use_sparse else 'DENSE'}")
        print(f"🎯 Support initial : {self.initial_support:.4f}")

        # ------------------------------
        # Étape 1 : itemsets de taille 1
        # ------------------------------
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
            'support': self.min_support
        })

        print(f"📊 Itération 1 : {len(frequent_1)} itemsets fréquents trouvés")

        # ------------------------------
        # Étapes suivantes
        # ------------------------------
        k = 2
        current_frequent = frequent_1

        while current_frequent:
            candidates = self._generate_candidates(current_frequent, k)
            if not candidates:
                break

            # Comptage des supports
            if self.use_sparse:
                candidate_counts = self._count_support_sparse(candidates)
            else:
                candidate_counts = defaultdict(int)
                for transaction in self.transactions:
                    tset = set(transaction)
                    for candidate in candidates:
                        if candidate.issubset(tset):
                            candidate_counts[candidate] += 1

            if not candidate_counts:
                break

            # ------------------------------
            # Recalcul dynamique du support
            # ------------------------------
            all_supports = np.array([c / self.n_transactions for c in candidate_counts.values()])
            mean_support = np.mean(all_supports)
            std_support = np.std(all_supports)

            # Filtrage des valeurs extrêmes
            filtered = [s for s in all_supports if abs(s - mean_support) <= 2 * std_support]
            adjusted_mean = np.mean(filtered) if len(filtered) > 0 else mean_support

            prev_support = self.min_support
            self.min_support = max(0.01, min(0.5, 0.5 * prev_support + 0.5 * adjusted_mean))

            self.support_history.append({
                'iteration': k,
                'support': self.min_support,
                'mean': adjusted_mean,
                'std': std_support
            })

            print(f"⚙️  Itération {k} → recalcul du support : moyenne={adjusted_mean:.4f}, σ={std_support:.4f} → nouveau min_support={self.min_support:.4f}")

            # ------------------------------
            # Sélection des nouveaux itemsets fréquents
            # ------------------------------
            new_frequent = {
                itemset: count / self.n_transactions
                for itemset, count in candidate_counts.items()
                if (count / self.n_transactions) >= self.min_support
            }

            if not new_frequent:
                break

            self.frequent_itemsets[k] = [
                {'itemset': list(itemset), 'support': sup}
                for itemset, sup in new_frequent.items()
            ]

            self.iteration_stats.append({
                'iteration': k,
                'candidates': len(candidates),
                'frequent': len(new_frequent),
                'support': self.min_support
            })

            print(f"📊 Itération {k} : {len(new_frequent)} itemsets fréquents trouvés")
            current_frequent = new_frequent
            k += 1

        self.execution_time = time.time() - start_time
        print(f"\n✅ Apriori terminé ({k - 1} itérations) en {self.execution_time:.2f}s")
        print(f"📦 Total itemsets fréquents : {sum(len(v) for v in self.frequent_itemsets.values())}")
        return self

    # ============================================================
    # SOUS-FONCTIONS
    # ============================================================

    def _prepare_matrix_representation(self):
        """Prépare une représentation matricielle (dense ou creuse)."""
        all_items = sorted({item for t in self.transactions for item in t})
        self.item_to_idx = {item: i for i, item in enumerate(all_items)}
        self.idx_to_item = {i: item for item, i in self.item_to_idx.items()}

        n_items = len(all_items)
        avg_len = np.mean([len(t) for t in self.transactions])
        density = avg_len / n_items if n_items > 0 else 0

        self.use_sparse = density < 0.3  # Seuil de densité
        print(f"📊 Statistiques matrice : items={n_items}, transactions={self.n_transactions}, "
              f"taille_moy={avg_len:.2f}, densité={density*100:.1f}%")

        if self.use_sparse:
            mat = lil_matrix((self.n_transactions, n_items), dtype=bool)
            for t_idx, transaction in enumerate(self.transactions):
                for item in transaction:
                    mat[t_idx, self.item_to_idx[item]] = True
            self.transaction_matrix = mat.tocsr()

    def _count_support_sparse(self, candidates):
        """Comptage optimisé du support avec matrice creuse."""
        candidate_counts = defaultdict(int)
        X = self.transaction_matrix

        for candidate in candidates:
            indices = [self.item_to_idx[item] for item in candidate]
            if len(indices) == 1:
                count = X[:, indices[0]].sum()
            else:
                mask = np.all(X[:, indices].toarray(), axis=1)
                count = np.sum(mask)
            candidate_counts[candidate] = int(count)

        return candidate_counts

    def calculate_adaptive_support(self):
        """Calcul initial adaptatif du support."""
        print("\n⚙️ Calcul du support minimal adaptatif initial")
        print("=" * 70)

        if not self.transactions:
            self.min_support = 0.05
            return self.min_support

        avg_len = np.mean([len(t) for t in self.transactions])
        unique_items = len({item for t in self.transactions for item in t})
        density = avg_len / unique_items if unique_items else 0

        base = 0.02
        size_factor = 0.15 if self.n_transactions < 100 else (0.08 if self.n_transactions < 500 else 0.03)
        density_factor = max(0.01, min(0.1, density * 0.5))
        support = max(base, min(0.3, size_factor + density_factor))

        self.min_support = support
        print(f"✅ Support initial calculé : {support:.4f} (densité={density:.2%})")
        return support

    def _generate_candidates(self, frequent_itemsets, k):
        """Génère les candidats de taille k."""
        candidates = set()
        itemsets = sorted([tuple(sorted(i)) for i in frequent_itemsets.keys()])

        for i in range(len(itemsets)):
            for j in range(i + 1, len(itemsets)):
                a, b = itemsets[i], itemsets[j]
                if a[:-1] == b[:-1]:
                    new = frozenset(a + (b[-1],))
                    if self._has_frequent_subsets(new, frequent_itemsets):
                        candidates.add(new)
        return candidates

    def _has_frequent_subsets(self, candidate, frequent_itemsets):
        """Vérifie la propriété antimonotone."""
        for item in candidate:
            subset = candidate - frozenset([item])
            if subset not in frequent_itemsets:
                return False
        return True

    # ============================================================
    # RÈGLES D’ASSOCIATION
    # ============================================================

    def _get_support(self, itemset):
        """Retourne le support d’un itemset."""
        k = len(itemset)
        if k in self.frequent_itemsets:
            for d in self.frequent_itemsets[k]:
                if frozenset(d['itemset']) == itemset:
                    return d['support']
        return 0

    def generate_rules(self):
        """Génère les règles d’association (avec lift et confiance)."""
        print("\n🎯 Génération des règles d’association")
        print("=" * 70)
        self.rules = []

        for k, itemsets in self.frequent_itemsets.items():
            if k < 2:
                continue
            for data in itemsets:
                itemset = frozenset(data['itemset'])
                s_itemset = data['support']

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

                        if confidence >= self.min_confidence:
                            self.rules.append({
                                'antecedent': list(antecedent),
                                'consequent': list(consequent),
                                'support': s_itemset,
                                'confidence': confidence,
                                'lift': lift
                            })

        self.rules.sort(key=lambda x: (-x['confidence'], -x['support']))
        print(f"✅ {len(self.rules)} règles générées.")
        return self.rules

    def get_statistics(self):
        """Retourne des statistiques détaillées d'exécution."""
        return {
            'execution_time': self.execution_time,
            'total_transactions': self.n_transactions,
            'total_rules': len(self.rules),
            'total_frequent_itemsets': sum(len(v) for v in self.frequent_itemsets.values()),
            'initial_support': self.initial_support,
            'final_support': self.min_support,
            'use_sparse_matrix': self.use_sparse,
            'support_history': self.support_history,
            'iteration_stats': self.iteration_stats,
            'itemsets_by_size': {k: len(v) for k, v in self.frequent_itemsets.items()}
        }
