import pandas as pd
import os
import numpy as np

class HeartFailureDataLoader:
    def __init__(self, dataset_path=None):
        # Chemin par défaut (utilise un chemin relatif plus portable)
        if dataset_path is None:
            # Essaie plusieurs chemins possibles
            possible_paths = [
                "data/datasets/heart.csv",
                "../data/datasets/heart.csv",
                "heart.csv",
                r"C:\Users\Maroua Cerine\OneDrive\Bureau\IA\apriori_project\data\datasets\heart.csv"
            ]
            self.dataset_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    self.dataset_path = path
                    break
            if self.dataset_path is None:
                self.dataset_path = possible_paths[0]  # Chemin par défaut
        else:
            self.dataset_path = dataset_path

    def load_dataset(self):
        """Charger et transformer le dataset médical"""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"❌ Fichier introuvable : {self.dataset_path}\n"
                f"Veuillez placer 'heart.csv' dans le répertoire approprié\n"
                f"Chemins recherchés: data/datasets/heart.csv ou même répertoire"
            )

        print(f"✅ Chargement du dataset : {self.dataset_path}")
        df = pd.read_csv(self.dataset_path)

        # Normaliser les noms de colonnes
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]
        
        print(f"📊 Dataset initial: {len(df)} lignes, {len(df.columns)} colonnes")
        
        # Afficher les colonnes détectées
        print(f"📋 Colonnes détectées: {', '.join(df.columns)}")
        
        # Gérer les valeurs manquantes avant discrétisation
        df = self._handle_missing_values(df)
        
        # Validation des données
        df = self._validate_data(df)

        # Discrétisation des variables continues
        df = self._discretize(df)

        # Transformation en liste de transactions
        transactions = []
        for _, row in df.iterrows():
            transaction = [f"{col.lower()}_{str(row[col]).replace(' ', '_')}" 
                          for col in df.columns if pd.notna(row[col])]
            transactions.append(transaction)

        print(f"📊 {len(transactions)} transactions générées (patients).")
        print(f"📦 {len(df.columns)} attributs par patient.\n")
        return transactions

    def _validate_data(self, df):
        """
        NOUVEAU: Valide et nettoie les données
        """
        print("\n🔍 Validation des données...")
        
        # Supprimer les lignes entièrement vides
        df = df.dropna(how='all')
        
        # Valider les colonnes numériques
        numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        for col in numeric_cols:
            if col in df.columns:
                # Supprimer les valeurs négatives inappropriées
                if col in ['Age', 'RestingBP', 'Cholesterol', 'MaxHR']:
                    df.loc[df[col] < 0, col] = np.nan
                
                # Supprimer les valeurs aberrantes extrêmes
                if col == 'Age':
                    df.loc[df[col] > 120, col] = np.nan
                elif col == 'RestingBP':
                    df.loc[(df[col] < 50) | (df[col] > 250), col] = np.nan
                elif col == 'Cholesterol':
                    df.loc[df[col] > 600, col] = np.nan
                elif col == 'MaxHR':
                    df.loc[(df[col] < 40) | (df[col] > 220), col] = np.nan
        
        print("✅ Validation terminée")
        return df

    def _discretize(self, df):
        """
        AMÉLIORATION: Discrétisation plus robuste avec gestion d'erreurs
        """
        print("\n🔄 Discrétisation des variables continues...")
        
        # Age
        if 'Age' in df.columns:
            try:
                df['Age'] = pd.cut(df['Age'],
                    bins=[0, 30, 40, 50, 60, 70, 80, 120],
                    labels=['<30', '30-40', '40-50', '50-60', '60-70', '70-80', '80+'],
                    include_lowest=True
                )
                print("   ✓ Age discrétisé")
            except Exception as e:
                print(f"   ⚠️  Erreur lors de la discrétisation de Age: {e}")

        # Pression artérielle au repos
        if 'RestingBP' in df.columns:
            try:
                df['RestingBP'] = pd.cut(df['RestingBP'],
                    bins=[0, 100, 120, 140, 160, 180, 250],
                    labels=['<100', '100-120', '120-140', '140-160', '160-180', '>180'],
                    include_lowest=True
                )
                print("   ✓ RestingBP discrétisé")
            except Exception as e:
                print(f"   ⚠️  Erreur lors de la discrétisation de RestingBP: {e}")

        # Cholestérol
        if 'Cholesterol' in df.columns:
            try:
                # CORRECTION: Gérer les valeurs 0 dans Cholesterol (souvent manquantes)
                df.loc[df['Cholesterol'] == 0, 'Cholesterol'] = np.nan
                df['Cholesterol'] = pd.cut(df['Cholesterol'],
                    bins=[0, 150, 200, 250, 300, 600],
                    labels=['<150', '150-200', '200-250', '250-300', '>300'],
                    include_lowest=True
                )
                print("   ✓ Cholesterol discrétisé")
            except Exception as e:
                print(f"   ⚠️  Erreur lors de la discrétisation de Cholesterol: {e}")
            
        # Fréquence cardiaque maximale
        if 'MaxHR' in df.columns:
            try:
                df['MaxHR'] = pd.cut(df['MaxHR'],
                    bins=[0, 80, 100, 120, 140, 160, 220],
                    labels=['<80', '80-100', '100-120', '120-140', '140-160', '>160'],
                    include_lowest=True
                )
                print("   ✓ MaxHR discrétisé")
            except Exception as e:
                print(f"   ⚠️  Erreur lors de la discrétisation de MaxHR: {e}")

        # Oldpeak (dépression ST)
        if 'Oldpeak' in df.columns:
            try:
                df['Oldpeak'] = pd.cut(df['Oldpeak'],
                    bins=[-2, 0, 1, 2, 4, 8],
                    labels=['≤0', '0-1', '1-2', '2-4', '>4'],
                    include_lowest=True
                )
                print("   ✓ Oldpeak discrétisé")
            except Exception as e:
                print(f"   ⚠️  Erreur lors de la discrétisation de Oldpeak: {e}")

        print("✅ Discrétisation terminée\n")
        return df

    def _handle_missing_values(self, df):
        """
         Gestion  des valeurs manquantes
        """
        print("\n🔧 Gestion des valeurs manquantes...")
        
        missing_counts = df.isnull().sum()
        if missing_counts.sum() == 0:
            print("   ✓ Aucune valeur manquante détectée")
            return df
        
        print(f"\n   Valeurs manquantes détectées:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"   - {col}: {count} ({count/len(df)*100:.1f}%)")
        
        # Pour les variables numériques : utiliser la médiane
        numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        
        for col in numeric_cols:
            if col in df.columns and df[col].isnull().any():
                median_value = df[col].median()
                missing_count = df[col].isnull().sum()
                df[col].fillna(median_value, inplace=True)
                print(f"   ✓ {col}: {missing_count} valeurs remplacées par médiane ({median_value:.1f})")
        
        # Pour les variables catégorielles : utiliser le mode
        categorical_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 
                           'ExerciseAngina', 'ST_Slope', 'HeartDisease']
        
        for col in categorical_cols:
            if col in df.columns and df[col].isnull().any():
                if len(df[col].mode()) > 0:
                    mode_value = df[col].mode()[0]
                    missing_count = df[col].isnull().sum()
                    df[col].fillna(mode_value, inplace=True)
                    print(f"   ✓ {col}: {missing_count} valeurs remplacées par mode ({mode_value})")

        print("✅ Valeurs manquantes traitées\n")
        return df

    def get_stats(self, transactions):
        """
        AMÉLIORATION: Statistiques plus détaillées
        """
        from collections import Counter

        all_items = [item for t in transactions for item in t]
        item_counts = Counter(all_items)

        print("="*70)
        print("📊 STATISTIQUES DU DATASET MÉDICAL")
        print("="*70)
        print(f"Nombre de patients : {len(transactions)}")
        print(f"Nombre d'attributs uniques : {len(item_counts)}")
        
        transaction_lengths = [len(t) for t in transactions]

        print(f"\n🏆 Top 5 attributs les plus fréquents :")
        print("-"*70)
        for i, (item, count) in enumerate(item_counts.most_common(5), 1):
            percentage = (count / len(transactions)) * 100
            print(f"{i:2d}. {item:35s} {count:5d} ({percentage:5.1f}%)")

        # Statistiques par catégorie
        print(f"\n📂 Répartition par catégorie d'attributs :")
        print("-"*70)
        categories = {}
        for item in item_counts.keys():
            category = item.split('_')[0]
            categories[category] = categories.get(category, 0) + 1
        
        for category, count in sorted(categories.items(), key=lambda x: -x[1]):
            print(f"   {category:20s} : {count} valeurs distinctes")

        print("="*70 + "\n")
    
    def save_transactions(self, transactions, filepath='transactions.csv'):
        """
        NOUVEAU: Sauvegarde les transactions dans un fichier
        """
        import csv
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Transaction_ID', 'Items'])
            for i, transaction in enumerate(transactions, 1):
                writer.writerow([i, ', '.join(transaction)])
        
        print(f"✅ Transactions sauvegardées dans: {filepath}")