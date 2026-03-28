
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt

# df = pd.DataFrame({
#     "Nom":["Omar", "Fatima", "Yassine", "Omar"],
#     "Genre":["Homme", "Femme", "Homme", "Homme"],
#     "Ville":["Casablanca", "Rabat", "Tanger", "Fès"],
#     "Age":[25,30,None,25],
#     "Salaire (MAD)":[8000,10000,7500,8000]
# })

# # ============================== Suppression des doublons
# df = df.drop_duplicates()

# # ============================== Encodage des Variables Catégorielles
# # One-hot encoding 0 or 1 (encodage binaire) de la colonne "Genre"
# encoder = LabelEncoder()
# df["Genre"] = encoder.fit_transform(df["Genre"])

# # Encodage on-hot de la colonne "Ville"
# encoder = OneHotEncoder(sparse_output=False)
# encoded = encoder.fit_transform(df[["Ville"]])

# # Convertir en DataFrame Pandas pour affichage
# df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Ville']))


# # ============================== Gestion des valeurs manquantes
# # ============ Suppression
# # Exemple de DataFrame avec valeurs manquantes
# data = {
#     'Nom': ['Ali', 'Samira', 'Youssef', 'Khadija'],
#     'Salaire': [7000, None, 9000, None],
#     'Ville': ['Rabat', 'Casablanca', None, 'Fès']
# }
# df = pd.DataFrame(data)

# # Supprimer les lignes avec au moins une valeurs manquante
# df_cleaned = df.dropna()


# # ============ Remplacement 
# data = {'Salaire':[5000, 7000, None, 8000]}
# df = pd.DataFrame(data)
# # ============ Remplacement Par moyenne
# imputer = SimpleImputer(strategy='mean')
# df[['Salaire']] = imputer.fit_transform(df[['Salaire']])

# # ============ Remplacement Par mediane
# imputer = SimpleImputer(strategy='median')
# df[['Salaire']] = imputer.fit_transform(df[['Salaire']])

# # ============ Remplacement Par valeur la plus fréquete pour les variables qualitatives
# data = {
#     'Nom':['Ali', 'Samira', 'Youssef', 'Khadija'],
#     'Ville':['Rabat', 'Casablanca', np.nan, 'Rabat']
# }
# imputer = SimpleImputer(strategy='most_frequent')
# df[['Ville']] = imputer.fit_transform(df[['Ville']])

# # ============================== Standarisation 
# data_std = {'Salaire':[5000, 7000, 8000, 9000, 10000]}
# df_std = pd.DataFrame(data_std)
# # Standarisation
# scaler = StandardScaler()
# df_std['Salaire_std'] = scaler.fit_transform(df_std[['Salaire']])

# ============================== Normalisation (Min-Max Scaling)

# Donnees avec unites differents
# Salaire (MAD), Age (annees), Taux de satisfaction (%)

data = pd.DataFrame({
    "Salaire (MAD)":[5000, 7000, 10000],
    "Age (anneed)":[25,35,45],
    "Satisfaction (%)":[70,80,90]
})

# Appliquer la normalisation
scaler = MinMaxScaler()
data_normalized = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

