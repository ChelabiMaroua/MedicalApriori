"""
Script de test pour l'API CardioAI
Teste tous les endpoints et vérifie la connexion backend-frontend
"""

import requests
import json
from typing import Dict, Any

BASE_URL = "http://127.0.0.1:8000"

def print_section(title: str):
    """Afficher une section formatée."""
    print("\n" + "=" * 70)
    print(f"🧪 {title}")
    print("=" * 70)

def test_health():
    """Test 1: Vérifier la santé du service."""
    print_section("TEST 1: Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Service actif: {data['service']} v{data['version']}")
            return True
        else:
            print(f"❌ Erreur: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Impossible de se connecter au serveur!")
        print("   Assurez-vous que le serveur est démarré avec: python main.py")
        return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_root():
    """Test 2: Tester l'endpoint racine."""
    print_section("TEST 2: Root Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Service: {data['service']}")
            print(f"📝 Description: {data['description']}")
            print(f"🔧 Features:")
            for feature in data['features']:
                print(f"   • {feature}")
            return True
        else:
            print(f"❌ Erreur: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_dataset_info():
    """Test 3: Récupérer les informations du dataset."""
    print_section("TEST 3: Dataset Info")
    
    try:
        response = requests.get(f"{BASE_URL}/dataset_info", timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Transactions: {data['total_transactions']}")
            print(f"📦 Items uniques: {data['total_unique_items']}")
            print(f"📊 Taille moyenne: {data['avg_transaction_length']:.2f}")
            
            print(f"\n🏆 Top 5 items:")
            for item in data['top_items'][:5]:
                print(f"   • {item['item']}: {item['count']} ({item['percentage']}%)")
            
            print(f"\n📂 Catégories:")
            for cat, count in list(data['categories'].items())[:5]:
                print(f"   • {cat}: {count} valeurs")
            
            return True
        else:
            print(f"❌ Erreur: {response.status_code}")
            print(f"Réponse: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_apriori_auto_support():
    """Test 4: Exécuter Apriori avec support automatique."""
    print_section("TEST 4: Apriori (Support Auto)")
    
    payload = {
        "min_support": None,  # Support automatique
        "min_confidence": 0.7
    }
    
    try:
        print(f"📤 Envoi de la requête: {payload}")
        response = requests.post(
            f"{BASE_URL}/run_apriori",
            json=payload,
            timeout=60
        )
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Succès: {data['success']}")
            print(f"⏱️  Temps d'exécution: {data['execution_time']:.2f}s")
            print(f"📊 Matrice utilisée: {data['matrix_type']}")
            print(f"📦 Transactions: {data['total_transactions']}")
            print(f"📋 Règles générées: {data['total_rules']}")
            
            stats = data['statistics']
            print(f"\n📈 Statistiques:")
            print(f"   • Support initial: {stats['initial_support']:.4f}")
            print(f"   • Support final: {stats['final_support']:.4f}")
            print(f"   • Variation: {stats['support_variation_pct']:+.1f}%")
            print(f"   • Itemsets fréquents: {stats['total_frequent_itemsets']}")
            print(f"   • Taux élagage moyen: {stats['avg_pruning_rate']*100:.1f}%")
            
            if data['total_rules'] > 0:
                print(f"\n🏆 Top 3 règles:")
                for i, rule in enumerate(data['rules'][:3], 1):
                    print(f"\n   {i}. {rule['antecedent']} → {rule['consequent']}")
                    print(f"      Support: {rule['support']:.3f}")
                    print(f"      Confiance: {rule['confidence']:.3f}")
                    print(f"      Lift: {rule['lift']:.2f}")
                    print(f"      Conviction: {rule['conviction']:.2f}")
            
            return True
        else:
            print(f"❌ Erreur: {response.status_code}")
            print(f"Réponse: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_apriori_custom_support():
    """Test 5: Exécuter Apriori avec support personnalisé."""
    print_section("TEST 5: Apriori (Support Personnalisé)")
    
    payload = {
        "min_support": 0.05,
        "min_confidence": 0.6
    }
    
    try:
        print(f"📤 Envoi de la requête: {payload}")
        response = requests.post(
            f"{BASE_URL}/run_apriori",
            json=payload,
            timeout=60
        )
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Succès")
            print(f"⏱️  Temps: {data['execution_time']:.2f}s")
            print(f"📋 Règles: {data['total_rules']}")
            print(f"📊 Matrice: {data['matrix_type']}")
            return True
        else:
            print(f"❌ Erreur: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_cors():
    """Test 6: Vérifier la configuration CORS."""
    print_section("TEST 6: Configuration CORS")
    
    headers = {
        'Origin': 'http://localhost:3000',
        'Access-Control-Request-Method': 'POST',
        'Access-Control-Request-Headers': 'Content-Type'
    }
    
    try:
        response = requests.options(
            f"{BASE_URL}/run_apriori",
            headers=headers,
            timeout=5
        )
        print(f"Status Code: {response.status_code}")
        
        cors_headers = {
            k: v for k, v in response.headers.items() 
            if k.lower().startswith('access-control')
        }
        
        if cors_headers:
            print("✅ CORS configuré:")
            for header, value in cors_headers.items():
                print(f"   • {header}: {value}")
            return True
        else:
            print("⚠️  Aucun header CORS détecté")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def run_all_tests():
    """Exécuter tous les tests."""
    print("\n" + "=" * 70)
    print("🚀 DÉMARRAGE DES TESTS DE L'API CARDIOAI")
    print("=" * 70)
    
    tests = [
        ("Health Check", test_health),
        ("Root Endpoint", test_root),
        ("Dataset Info", test_dataset_info),
        ("Apriori Auto", test_apriori_auto_support),
        ("Apriori Custom", test_apriori_custom_support),
        ("CORS Config", test_cors)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except KeyboardInterrupt:
            print("\n⚠️  Tests interrompus par l'utilisateur")
            break
        except Exception as e:
            print(f"❌ Erreur inattendue dans {name}: {e}")
            results.append((name, False))
    
    # Résumé
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\n🎯 Score: {passed}/{total} tests réussis ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("🎉 Tous les tests sont passés avec succès!")
    else:
        print("⚠️  Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")
    
    return passed == total

if __name__ == "__main__":
    import sys
    
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                   SCRIPT DE TEST CARDIOAI API                     ║
║                                                                   ║
║  Ce script teste tous les endpoints de l'API et vérifie la       ║
║  connexion entre le backend et le potentiel frontend.            ║
║                                                                   ║
║  Prérequis:                                                       ║
║  1. Le serveur doit être démarré: python main.py                 ║
║  2. Le dataset heart.csv doit être présent                       ║
║                                                                   ║
║  Appuyez sur Ctrl+C pour arrêter à tout moment                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        input("Appuyez sur Entrée pour démarrer les tests...")
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests annulés par l'utilisateur")
        sys.exit(1)