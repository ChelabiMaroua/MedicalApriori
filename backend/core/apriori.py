from collections import defaultdict
from itertools import combinations
import numpy as np

class Apriori:
    def __init__(self, min_support=None, min_confidence=0.6):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.transactions = []
        self.frequent_itemsets = {}
        self.rules = []
        self.n_transactions = 0
        
    def fit(self, transactions):
        """Exécuter l'algorithme Apriori"""
        self.transactions = transactions
        self.n_transactions = len(transactions)
        
        if self.n_transactions == 0:
            raise ValueError("❌ Aucune transaction fournie")
        
        # Calcul automatique du support si non défini
        if self.min_support is None:
            self.calculate_adaptive_support()
        
        print(f"\n🔍 Démarrage Apriori")
        print("="*70)
        
        # Étape 1 : itemsets de taille 1
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
            print("⚠️  Aucun itemset fréquent trouvé. Essayez de réduire min_support.")
            return self
        
        self.frequent_itemsets[1] = [
            {'itemset': list(itemset), 'support': sup}
            for itemset, sup in frequent_1.items()
        ]
        
        print(f"📊 Itération 1: {len(frequent_1)} itemsets fréquents trouvés")
        
        # Étapes suivantes
        k = 2
        current_frequent = frequent_1
        while current_frequent:
            candidates = self._generate_candidates(current_frequent, k)
            if not candidates:
                break
            
            candidate_counts = defaultdict(int)
            for transaction in self.transactions:
                tset = set(transaction)
                for candidate in candidates:
                    if candidate.issubset(tset):
                        candidate_counts[candidate] += 1
            
            # 🔥 On calcule tous les supports bruts à cette itération
            all_supports = np.array([count / self.n_transactions for count in candidate_counts.values()])
            
            # 🔥 Recalcul du min_support dynamique selon la distribution actuelle
            if len(all_supports) > 0:
                mean_support = np.mean(all_supports)
                std_support = np.std(all_supports)
                
                # On écarte les valeurs extrêmes (au-delà de 2 écarts-types)
                filtered_supports = [s for s in all_supports if abs(s - mean_support) <= 2 * std_support]
                
                if len(filtered_supports) > 0:
                    adjusted_mean = np.mean(filtered_supports)
                else:
                    adjusted_mean = mean_support
                
                # On ajuste le min_support avec un facteur de stabilité
                prev_support = self.min_support
                self.min_support = max(0.01, min(0.5, 0.5 * prev_support + 0.5 * adjusted_mean))
                
                print(f"⚙️  Support recalculé à l’itération {k}: moyenne={adjusted_mean:.4f}, σ={std_support:.4f} → min_support={self.min_support:.4f}")
            
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
            
            print(f"📊 Itération {k}: {len(new_frequent)} itemsets fréquents trouvés")
            current_frequent = new_frequent
            k += 1
        
        print(f"\n✅ Apriori terminé ({k-1} itérations)")
        print(f"📦 Total itemsets fréquents: {sum(len(v) for v in self.frequent_itemsets.values())}")
        return self

    def calculate_adaptive_support(self):
        """Calcul initial adaptatif du support basé sur les caractéristiques du dataset"""
        print("\n⚙️ Calcul du support minimal adaptatif initial")
        print("="*70)
        
        avg_length = sum(len(t) for t in self.transactions) / len(self.transactions)
        unique_items = len(set(item for t in self.transactions for item in t))
        density = avg_length / unique_items if unique_items > 0 else 0
        
        # Heuristique initiale
        base_support = 0.02
        if self.n_transactions < 100:
            size_factor = 0.15
        elif self.n_transactions < 500:
            size_factor = 0.08
        else:
            size_factor = 0.03
        density_factor = max(0.01, min(0.1, density * 0.5))
        support = max(base_support, min(0.3, size_factor + density_factor))
        
        self.min_support = support
        print(f"✅ Support initial calculé: {support:.4f}")
        return support

    # Autres méthodes (_generate_candidates, _has_frequent_subsets, _get_support, generate_rules, analyze_results, etc.)
    # inchangées
    
    def _generate_candidates(self, frequent_itemsets, k):
        """Génère les candidats de taille k selon Apriori"""
        candidates = set()
        itemsets_list = sorted([tuple(sorted(itemset)) for itemset in frequent_itemsets.keys()])
        
        for i in range(len(itemsets_list)):
            for j in range(i+1, len(itemsets_list)):
                itemset1 = itemsets_list[i]
                itemset2 = itemsets_list[j]
                # Vérification du préfixe commun
                if itemset1[:-1] == itemset2[:-1]:
                    new_candidate = frozenset(itemset1 + (itemset2[-1],))
                    # Élagage basé sur la propriété antimonotone
                    if self._has_frequent_subsets(new_candidate, frequent_itemsets, k):
                        candidates.add(new_candidate)
        return candidates
    
    def _has_frequent_subsets(self, candidate, frequent_itemsets, k):
        """Vérifie que tous les sous-ensembles de taille k-1 sont fréquents"""
        for item in candidate:
            subset = candidate - frozenset([item])
            if subset not in frequent_itemsets:
                return False
        return True
    
    def _get_support(self, itemset):
        """Retourne le support d'un itemset"""
        k = len(itemset)
        if k in self.frequent_itemsets:
            for data in self.frequent_itemsets[k]:
                if frozenset(data['itemset']) == itemset:
                    return data['support']
        return 0
    
    def generate_rules(self):
        """Génère les règles d'association avec support et confiance"""
        print("\n🎯 Génération des règles d'association")
        print("="*70)
        self.rules = []
        
        for k, itemsets in self.frequent_itemsets.items():
            if k < 2:
                continue
            for data in itemsets:
                itemset = frozenset(data['itemset'])
                itemset_support = data['support']
                
                # Génération de toutes les règles possibles
                for i in range(1, len(itemset)):
                    for antecedent in combinations(itemset, i):
                        antecedent = frozenset(antecedent)
                        consequent = itemset - antecedent
                        antecedent_support = self._get_support(antecedent)
                        
                        if antecedent_support == 0:
                            continue
                        
                        confidence = itemset_support / antecedent_support
                        
                        if confidence >= self.min_confidence:
                            self.rules.append({
                                'antecedent': list(antecedent),
                                'consequent': list(consequent),
                                'support': itemset_support,
                                'confidence': confidence
                            })
        
        # Tri par confiance puis support
        self.rules.sort(key=lambda x: (-x['confidence'], -x['support']))
        print(f"✅ {len(self.rules)} règles générées")
        return self.rules
    
    def analyze_results(self, top_n=5):
        """Analyse détaillée des résultats avec support et confiance"""
        print("\n" + "="*70)
        print(f"📋 ANALYSE DES RÉSULTATS (TOP {top_n})")
        print("="*70)
        
        if not self.rules:
            print("❌ Aucune règle trouvée.")
            return
        
        # TOP N des règles par confiance
        print(f"\n🥇 TOP {top_n} RÈGLES PAR CONFIANCE :")
        print("-"*70)
        for i, rule in enumerate(self.rules[:top_n], 1):
            self._print_rule(rule, i)
        
        # TOP N des règles par support
        print(f"\n📊 TOP {top_n} RÈGLES PAR SUPPORT :")
        print("-"*70)
        sorted_by_support = sorted(self.rules, key=lambda r: -r['support'])
        for i, rule in enumerate(sorted_by_support[:top_n], 1):
            self._print_rule(rule, i)
        
        # Statistiques globales
        print("\n📊 STATISTIQUES GLOBALES :")
        print("-"*70)
        confidences = [r['confidence'] for r in self.rules]
        supports = [r['support'] for r in self.rules]
        
        print(f"Nombre total de règles: {len(self.rules)}")
        print(f"Confiance moyenne: {np.mean(confidences):.3f}")
        print(f"Confiance médiane: {np.median(confidences):.3f}")
        print(f"Confiance min/max: {min(confidences):.3f}/{max(confidences):.3f}")
        print(f"Support moyen: {np.mean(supports):.3f}")
        print(f"Support médian: {np.median(supports):.3f}")
        print(f"Support min/max: {min(supports):.3f}/{max(supports):.3f}")
    
    def _print_rule(self, rule, index):
        """Affichage formaté d'une règle"""
        antecedent = ' ET '.join(sorted(rule['antecedent']))
        consequent = ' ET '.join(sorted(rule['consequent']))
        
        print(f"\n📌 RÈGLE #{index}")
        print(f"   SI [{antecedent}]")
        print(f"   → ALORS [{consequent}]")
        print(f"   📊 Support={rule['support']:.3f} | Confiance={rule['confidence']:.3f}")
    
    def export_results(self, filepath='apriori_results.csv'):
        """Exporte les résultats vers un fichier CSV"""
        import pandas as pd
        
        if not self.rules:
            print("⚠️  Aucune règle à exporter")
            return
        
        export_data = []
        for rule in self.rules:
            export_data.append({
                'Antecedent': ' & '.join(sorted(rule['antecedent'])),
                'Consequent': ' & '.join(sorted(rule['consequent'])),
                'Support': rule['support'],
                'Confidence': rule['confidence']
            })
        
        df = pd.DataFrame(export_data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"✅ Résultats exportés vers: {filepath}")