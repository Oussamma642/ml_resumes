
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# ================= Étape 1 : créer les donnée =================
data = {
    "Genre":["Femme", "Homme", "Femme", "Homme"],
    "Age":[25,30,22,35],
    "Salaire":[3000,4000,2800,5000]
}
df = pd.DataFrame(data)

# ================= Étape 2 : Encodage =================
encoder = LabelEncoder()
df["Genre"] = encoder.fit_transform(df['Genre'])

# ================= Étape 3 : Standarisation =================
scalar = StandardScaler()
df[["Age", "Salaire"]] = scalar.fit_transform(df[["Age", "Salaire"]])

# ================= Étape 4 : Afficher les resultats =================
# print(df)

