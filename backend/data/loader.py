import pandas as pd
import numpy as np
from typing import List, Optional
import os


class HeartFailureDataLoader:
    """
    Chargeur de données pour l'analyse de défaillance cardiaque.
    Supporte à la fois les transactions (Apriori) et les données numériques (K-Means).
    """
    
    def __init__(self, filepath: str = "data/heart_failure_clinical_records_dataset.csv"):
        self.filepath = filepath
        self.df: Optional[pd.DataFrame] = None
        
    def load_dataset(self) -> List[List[str]]:
        """
        Charge le dataset et le transforme en transactions pour Apriori.
        
        Returns:
            Liste de transactions (chaque transaction est une liste d'items)
        """
        if not os.path.exists(self.filepath):
            print(f"⚠️ Fichier {self.filepath} non trouvé. Génération de données synthétiques...")
            return self._generate_synthetic_data()
        
        try:
            self.df = pd.read_csv(self.filepath)
            print(f"✅ Dataset chargé: {len(self.df)} enregistrements")
            
            transactions = []
            
            for _, row in self.df.iterrows():
                transaction = []
                
                # Transformation des colonnes en items binaires/catégoriels
                for col in self.df.columns:
                    value = row[col]
                    
                    # Gestion des valeurs numériques -> discrétisation
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        if col in ['age', 'creatinine_phosphokinase', 'ejection_fraction', 
                                   'platelets', 'serum_creatinine', 'serum_sodium', 'time']:
                            # Discrétisation en catégories
                            item = self._discretize_value(col, value)
                            if item:
                                transaction.append(item)
                    else:
                        # Colonnes catégorielles ou binaires
                        if value == 1 or str(value).lower() == 'true':
                            transaction.append(f"{col}_yes")
                        elif value == 0 or str(value).lower() == 'false':
                            transaction.append(f"{col}_no")
                        else:
                            transaction.append(f"{col}_{value}")
                
                if transaction:
                    transactions.append(transaction)
            
            return transactions
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            return self._generate_synthetic_data()
    
    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Retourne le DataFrame brut pour K-Means.
        """
        if self.df is None:
            if os.path.exists(self.filepath):
                self.df = pd.read_csv(self.filepath)
            else:
                # Génération de données synthétiques
                self.df = self._generate_synthetic_dataframe()
        
        return self.df
    
    def _discretize_value(self, column: str, value: float) -> Optional[str]:
        """
        Discrétise une valeur numérique en catégorie.
        """
        if pd.isna(value):
            return None
        
        # Règles de discrétisation par colonne
        discretization_rules = {
            'age': [(40, 'young'), (60, 'middle'), (float('inf'), 'senior')],
            'ejection_fraction': [(30, 'low'), (50, 'normal'), (float('inf'), 'high')],
            'serum_creatinine': [(1.0, 'normal'), (1.5, 'elevated'), (float('inf'), 'high')],
            'serum_sodium': [(135, 'low'), (145, 'normal'), (float('inf'), 'high')],
            'creatinine_phosphokinase': [(200, 'normal'), (500, 'elevated'), (float('inf'), 'very_high')],
            'platelets': [(150000, 'low'), (400000, 'normal'), (float('inf'), 'high')],
            'time': [(100, 'short'), (200, 'medium'), (float('inf'), 'long')]
        }
        
        if column not in discretization_rules:
            return None
        
        for threshold, category in discretization_rules[column]:
            if value <= threshold:
                return f"{column}_{category}"
        
        return None
    
    def _generate_synthetic_data(self) -> List[List[str]]:
        """
        Génère des données synthétiques pour tests.
        """
        print("🔧 Génération de 300 patients synthétiques...")
        
        np.random.seed(42)
        transactions = []
        
        for _ in range(300):
            transaction = []
            
            # Âge
            age = np.random.randint(40, 95)
            if age < 60:
                transaction.append("age_young")
            elif age < 75:
                transaction.append("age_middle")
            else:
                transaction.append("age_senior")
            
            # Anémie
            if np.random.rand() < 0.4:
                transaction.append("anaemia_yes")
            
            # Diabète
            if np.random.rand() < 0.4:
                transaction.append("diabetes_yes")
            
            # Hypertension
            if np.random.rand() < 0.35:
                transaction.append("high_blood_pressure_yes")
            
            # Fraction d'éjection
            ef = np.random.randint(15, 80)
            if ef < 30:
                transaction.append("ejection_fraction_low")
            elif ef < 50:
                transaction.append("ejection_fraction_normal")
            else:
                transaction.append("ejection_fraction_high")
            
            # Créatinine sérique
            creat = np.random.uniform(0.5, 9.0)
            if creat < 1.0:
                transaction.append("serum_creatinine_normal")
            elif creat < 1.5:
                transaction.append("serum_creatinine_elevated")
            else:
                transaction.append("serum_creatinine_high")
            
            # Sodium sérique
            sodium = np.random.randint(113, 148)
            if sodium < 135:
                transaction.append("serum_sodium_low")
            elif sodium < 145:
                transaction.append("serum_sodium_normal")
            else:
                transaction.append("serum_sodium_high")
            
            # Fumeur
            if np.random.rand() < 0.32:
                transaction.append("smoking_yes")
            
            # Décès
            if np.random.rand() < 0.32:
                transaction.append("DEATH_EVENT_yes")
            
            transactions.append(transaction)
        
        return transactions
    
    def _generate_synthetic_dataframe(self) -> pd.DataFrame:
        """
        Génère un DataFrame synthétique pour K-Means.
        """
        np.random.seed(42)
        n_samples = 300
        
        data = {
            'age': np.random.randint(40, 95, n_samples),
            'anaemia': np.random.binomial(1, 0.4, n_samples),
            'creatinine_phosphokinase': np.random.randint(23, 7861, n_samples),
            'diabetes': np.random.binomial(1, 0.4, n_samples),
            'ejection_fraction': np.random.randint(14, 80, n_samples),
            'high_blood_pressure': np.random.binomial(1, 0.35, n_samples),
            'platelets': np.random.uniform(25000, 850000, n_samples),
            'serum_creatinine': np.random.uniform(0.5, 9.4, n_samples),
            'serum_sodium': np.random.randint(113, 148, n_samples),
            'sex': np.random.binomial(1, 0.65, n_samples),
            'smoking': np.random.binomial(1, 0.32, n_samples),
            'time': np.random.randint(4, 285, n_samples),
            'DEATH_EVENT': np.random.binomial(1, 0.32, n_samples)
        }
        
        return pd.DataFrame(data)