import pandas as pd
import os
import numpy as np
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HeartFailureDataLoader:
    def __init__(self, dataset_path=None):
        if dataset_path is None:
            possible_paths = [
                "data/datasets/heart.csv",
                "../data/datasets/heart.csv",
                "../../data/datasets/heart.csv",
                "heart.csv",
                "backend/data/datasets/heart.csv",
                os.path.join(os.path.dirname(__file__), "datasets", "heart.csv")
            ]
            self.dataset_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    self.dataset_path = path
                    break
            if self.dataset_path is None:
                self.dataset_path = possible_paths[0]
        else:
            self.dataset_path = dataset_path

    def load_dataset(self) -> List[List[str]]:
        """Charger et transformer le dataset médical en transactions."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"❌ Fichier introuvable : {self.dataset_path}\n"
                f"Veuillez placer 'heart.csv' dans un des répertoires suivants:\n"
                f"  • data/datasets/heart.csv\n"
                f"  • backend/data/datasets/heart.csv\n"
                f"  • Répertoire courant"
            )

        logger.info(f"✅ Chargement du dataset : {self.dataset_path}")
        df = pd.read_csv(self.dataset_path)

        # Normaliser les noms de colonnes
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]
        
        logger.info(f"📊 Dataset initial: {len(df)} lignes, {len(df.columns)} colonnes")
        logger.info(f"📋 Colonnes: {', '.join(df.columns)}")
        
        # Pipeline de traitement
        df = self._handle_missing_values(df)
        df = self._validate_data(df)
        df = self._discretize(df)
        
        # Transformation en transactions
        transactions = self._to_transactions(df)
        
        logger.info(f"📊 {len(transactions)} transactions générées")
        logger.info(f"📦 ~{np.mean([len(t) for t in transactions]):.1f} items par transaction\n")
        
        return transactions

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validation et nettoyage des données."""
        logger.info("🔍 Validation des données...")
        
        initial_rows = len(df)
        df = df.dropna(how='all')
        dropped = initial_rows - len(df)
        if dropped > 0:
            logger.info(f"   ✓ {dropped} lignes vides supprimées")
        
        # Validation des colonnes numériques
        validation_rules = {
            'Age': (0, 120),
            'RestingBP': (50, 250),
            'Cholesterol': (0, 600),
            'MaxHR': (40, 220),
            'Oldpeak': (-5, 10)
        }
        
        for col, (min_val, max_val) in validation_rules.items():
            if col in df.columns:
                invalid_mask = (df[col] < min_val) | (df[col] > max_val)
                invalid_count = invalid_mask.sum()
                if invalid_count > 0:
                    df.loc[invalid_mask, col] = np.nan
                    logger.info(f"   ⚠️  {col}: {invalid_count} valeurs aberrantes → NaN")
        
        logger.info("✅ Validation terminée")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gestion intelligente des valeurs manquantes."""
        logger.info("🔧 Gestion des valeurs manquantes...")
        
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing == 0:
            logger.info("   ✓ Aucune valeur manquante détectée")
            return df
        
        logger.info(f"   📊 Total de valeurs manquantes: {total_missing}")
        
        for col, count in missing_counts[missing_counts > 0].items():
            pct = count / len(df) * 100
            logger.info(f"   • {col}: {count} ({pct:.1f}%)")
        
        # Traitement spécial pour Cholesterol (0 = manquant)
        if 'Cholesterol' in df.columns:
            zero_count = (df['Cholesterol'] == 0).sum()
            if zero_count > 0:
                df.loc[df['Cholesterol'] == 0, 'Cholesterol'] = np.nan
                logger.info(f"   ⚠️  Cholesterol: {zero_count} valeurs '0' → NaN")
        
        # Imputation: médiane pour numériques, mode pour catégoriels
        numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        categorical_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 
                           'ExerciseAngina', 'ST_Slope', 'HeartDisease']
        
        for col in numeric_cols:
            if col in df.columns and df[col].isnull().any():
                median = df[col].median()
                count = df[col].isnull().sum()
                df[col].fillna(median, inplace=True)
                logger.info(f"   ✓ {col}: imputation médiane = {median:.1f}")
        
        for col in categorical_cols:
            if col in df.columns and df[col].isnull().any():
                if len(df[col].mode()) > 0:
                    mode = df[col].mode()[0]
                    count = df[col].isnull().sum()
                    df[col].fillna(mode, inplace=True)
                    logger.info(f"   ✓ {col}: imputation mode = {mode}")
        
        logger.info("✅ Valeurs manquantes traitées\n")
        return df

    def _discretize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Discrétisation robuste des variables continues."""
        logger.info("🔄 Discrétisation des variables continues...")
        
        discretization_rules = {
            'Age': {
                'bins': [0, 30, 40, 50, 60, 70, 80, 120],
                'labels': ['<30', '30-40', '40-50', '50-60', '60-70', '70-80', '80+']
            },
            'RestingBP': {
                'bins': [0, 100, 120, 140, 160, 180, 250],
                'labels': ['<100', '100-120', '120-140', '140-160', '160-180', '>180']
            },
            'Cholesterol': {
                'bins': [0, 150, 200, 250, 300, 600],
                'labels': ['<150', '150-200', '200-250', '250-300', '>300']
            },
            'MaxHR': {
                'bins': [0, 80, 100, 120, 140, 160, 220],
                'labels': ['<80', '80-100', '100-120', '120-140', '140-160', '>160']
            },
            'Oldpeak': {
                'bins': [-5, 0, 1, 2, 4, 10],
                'labels': ['≤0', '0-1', '1-2', '2-4', '>4']
            }
        }
        
        for col, rules in discretization_rules.items():
            if col in df.columns:
                try:
                    df[col] = pd.cut(
                        df[col],
                        bins=rules['bins'],
                        labels=rules['labels'],
                        include_lowest=True
                    )
                    logger.info(f"   ✓ {col} discrétisé")
                except Exception as e:
                    logger.error(f"   ✗ Erreur {col}: {e}")
        
        logger.info("✅ Discrétisation terminée\n")
        return df

    def _to_transactions(self, df: pd.DataFrame) -> List[List[str]]:
        """Conversion du DataFrame en liste de transactions."""
        transactions = []
        for _, row in df.iterrows():
            transaction = [
                f"{col.lower()}_{str(row[col]).replace(' ', '_').replace('.', '_')}"
                for col in df.columns 
                if pd.notna(row[col])
            ]
            transactions.append(transaction)
        return transactions

    def get_stats(self, transactions: List[List[str]]):
        """Afficher des statistiques détaillées sur les transactions."""
        from collections import Counter

        all_items = [item for t in transactions for item in t]
        item_counts = Counter(all_items)
        transaction_lengths = [len(t) for t in transactions]

        print("=" * 70)
        print("📊 STATISTIQUES DU DATASET")
        print("=" * 70)
        print(f"Patients (transactions): {len(transactions)}")
        print(f"Attributs uniques: {len(item_counts)}")
        print(f"Taille transaction: min={min(transaction_lengths)}, "
              f"max={max(transaction_lengths)}, "
              f"moyenne={np.mean(transaction_lengths):.2f}")
        
        print(f"\n🏆 Top 10 attributs les plus fréquents:")
        print("-" * 70)
        for i, (item, count) in enumerate(item_counts.most_common(10), 1):
            pct = count / len(transactions) * 100
            print(f"{i:2d}. {item:40s} {count:5d} ({pct:5.1f}%)")
        
        # Répartition par catégories
        categories = {}
        for item in item_counts.keys():
            cat = item.split('_')[0]
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"\n📂 Répartition par catégorie:")
        print("-" * 70)
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            print(f"   {cat:20s} : {count:3d} valeurs distinctes")
        
        print("=" * 70 + "\n")

    def save_transactions(self, transactions: List[List[str]], filepath='transactions.csv'):
        """Sauvegarder les transactions dans un fichier CSV."""
        import csv
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Transaction_ID', 'Item_Count', 'Items'])
            for i, transaction in enumerate(transactions, 1):
                writer.writerow([i, len(transaction), ', '.join(transaction)])
        
        logger.info(f"✅ Transactions sauvegardées: {filepath}")