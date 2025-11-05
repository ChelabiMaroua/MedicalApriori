import pandas as pd
import os
import sys
from core.apriori import Apriori
from data.loader import HeartFailureDataLoader

def main_heart_failure_analysis():
    """
    Analyse complète avec gestion d'erreurs et options configurables
    """
    print("🫀 ANALYSE APRIORI - MALADIES CARDIAQUES")
    print("=" * 70)
    print("Support adaptatif calculé automatiquement")
    print("=" * 70)

    try:
        # Configuration
        MIN_CONFIDENCE = 0.7
        TOP_N = 5
        EXPORT_RESULTS = True
        
        # 1️⃣ Chargement des données
        print("\n1️⃣  CHARGEMENT DU DATASET")
        print("-" * 70)
        
        # Permettre de spécifier un chemin personnalisé
        custom_path = None
        if len(sys.argv) > 1:
            custom_path = sys.argv[1]
            print(f"📂 Utilisation du chemin personnalisé: {custom_path}")
        
        data_loader = HeartFailureDataLoader(custom_path)
        transactions = data_loader.load_dataset()
        
        # Vérification des transactions
        if not transactions:
            raise ValueError("❌ Aucune transaction générée. Vérifiez le dataset.")
        
        data_loader.get_stats(transactions)
        
        # Option: sauvegarder les transactions
        if EXPORT_RESULTS:
            data_loader.save_transactions(transactions)

        # 2️⃣ Initialisation d'Apriori
        print("\n2️⃣  INITIALISATION D'APRIORI")
        print("-" * 70)
        print(f"⚙️  Paramètres:")
        print(f"   - min_confidence: {MIN_CONFIDENCE}")
        print(f"   - min_support: automatique (adaptatif)")
        
        apriori = Apriori(min_confidence=MIN_CONFIDENCE)

        # 3️⃣ Exécution d'Apriori
        print("\n3️⃣  EXÉCUTION DE L'ALGORITHME APRIORI")
        print("-" * 70)
        apriori.fit(transactions)
        
        # Vérification des itemsets fréquents
        if not apriori.frequent_itemsets:
            print("\n⚠️  ATTENTION: Aucun itemset fréquent trouvé!")
            return

        # 4️⃣ Génération des règles
        print("\n4️⃣  GÉNÉRATION DES RÈGLES D'ASSOCIATION")
        print("-" * 70)
        rules = apriori.generate_rules()
        
        if not rules:
            print("\n⚠️  ATTENTION: Aucune règle générée!")
            
            # Afficher les itemsets fréquents trouvés
            print(f"\n📦 Itemsets fréquents trouvés par taille:")
            for k, items in apriori.frequent_itemsets.items():
                print(f"   - Taille {k}: {len(items)} itemsets")
            return

        # 5️⃣ Analyse des résultats
        print("\n5️⃣  ANALYSE DES RÉSULTATS")
        print("-" * 70)
        apriori.analyze_results(top_n=TOP_N)
        
        # 6️⃣ Export des résultats
        if EXPORT_RESULTS:
            print("\n6️⃣  EXPORT DES RÉSULTATS")
            print("-" * 70)
            apriori.export_results('heart_apriori_results.csv')
            print("✅ Résultats exportés avec succès")
        
        # 7️⃣ Analyse spécifique pour le contexte médical
        print("\n7️⃣  ANALYSE MÉDICALE SPÉCIFIQUE")
        print("-" * 70)
        analyze_medical_patterns(apriori.rules)

        print("\n" + "="*70)
        print("✅ ANALYSE TERMINÉE AVEC SUCCÈS")
        print("="*70)
        print(f"\n📊 Résumé:")
        print(f"   - Transactions analysées: {len(transactions)}")
        print(f"   - Itemsets fréquents: {sum(len(v) for v in apriori.frequent_itemsets.values())}")
        print(f"   - Règles générées: {len(rules)}")
        print(f"   - Support utilisé: {apriori.min_support:.4f}")
        print(f"   - Confiance minimale: {MIN_CONFIDENCE}")

    except FileNotFoundError as e:
        print(f"\n❌ ERREUR DE FICHIER: {e}")
        
    except ValueError as e:
        print(f"\n❌ ERREUR DE DONNÉES: {e}")
        
    except Exception as e:
        print(f"\n❌ ERREUR INATTENDUE: {e}")
        import traceback
        print("\n📋 Détails de l'erreur:")
        traceback.print_exc()


def analyze_medical_patterns(rules):
    """Analyse spécifique des patterns médicaux avec support et confiance"""
    if not rules:
        print("   ❌ Aucune règle à analyser")
        return
    
    # Identifier les règles liées à HeartDisease
    heart_disease_rules = [
        r for r in rules 
        if any('heartdisease' in item.lower() for item in r['consequent'])
    ]
    
    if heart_disease_rules:
        print(f"\n❤️  Règles prédictives de maladie cardiaque ({len(heart_disease_rules)} trouvées):")
        print("-"*70)
        for i, rule in enumerate(heart_disease_rules[:5], 1):
            antecedent = ' ET '.join(sorted(rule['antecedent']))
            print(f"\n{i}. Facteurs de risque identifiés:")
            print(f"   {antecedent}")
            print(f"   → Probabilité de maladie: {rule['confidence']*100:.1f}%")
            print(f"   → Support: {rule['support']:.3f}")
    
    # Identifier les règles avec support élevé (patterns fréquents)
    frequent_patterns = sorted(rules, key=lambda x: -x['support'])[:5]
    print(f"\n🔍 Patterns les plus fréquents (top 5):")
    print("-"*70)
    for i, rule in enumerate(frequent_patterns, 1):
        ant = ' & '.join(sorted(rule['antecedent']))
        cons = ' & '.join(sorted(rule['consequent']))
        print(f"{i}. {ant} → {cons}")
        print(f"   Support: {rule['support']:.3f} | Confiance: {rule['confidence']:.3f}")
    
    # Statistiques par catégorie
    print(f"\n📊 Distribution des règles par catégorie:")
    print("-"*70)
    categories = {}
    for rule in rules:
        for item in rule['consequent']:
            category = item.split('_')[0]
            categories[category] = categories.get(category, 0) + 1
    
    for category, count in sorted(categories.items(), key=lambda x: -x[1])[:10]:
        print(f"   {category:20s}: {count} règles")


if __name__ == "__main__":
    main_heart_failure_analysis()