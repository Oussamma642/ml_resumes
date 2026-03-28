
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# Creation d'un dataframe avec valeurs manquantes
data = {
    "Age":[25,30,None, 45],
    "Salaire (MAD)":[7000, None, 8000, 12000]
}

df = pd.DataFrame(data)

# Etape[1]: Remplacer les valeurs manquantes par la moyenne
imputer = SimpleImputer()
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Etape[2]: Appliquer le PCA
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_imputed)

print(df_pca)
