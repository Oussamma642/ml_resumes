# Séance 5 : La Régression Linéaire

## **Dossier :** `[[regression-lineaire]]` **Parent :** `[[seance-5-entrainement-du-model]]` **Mots-clés :** #machine-learning #apprentissage-supervisé #régression #prédiction

## 🎯 1. Objectif Fondamental

La Régression Linéaire est un algorithme d'**apprentissage supervisé**.

> [!cite] Définition Son but est de prédire une valeur numérique continue (la variable cible $y$) à partir d'une ou plusieurs variables explicatives (les _features_ $x$). Pour que le modèle apprenne, il cherche à minimiser l'**Erreur Quadratique Moyenne (MSE)**, c'est-à-dire la distance entre ses prédictions et la réalité : $$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_{réel} - y_{prédit})^2$$

---

## 📏 2. La Régression Linéaire Simple

On l'utilise quand on n'a qu'**une seule variable explicative** pour faire notre prédiction. L'algorithme cherche à tracer la meilleure ligne droite possible à travers les données.

### L'Équation

$$y = ax + b$$

- **$y$** : La valeur prédite (ex: Prix d'une maison).
- **$x$** : La variable d'entrée (ex: Surface en $m^2$).
- **$a$** : La **Pente** ou **Poids** (_Weight_). Elle représente l'impact de $x$ sur $y$. Pour chaque mètre carré supplémentaire, le prix augmente de $a$.
- **$b$** : L'**Ordonnée à l'origine** ou **Biais** (_Bias_). C'est la valeur théorique de $y$ quand $x=0$.

---

## 📐 3. La Régression Linéaire Multiple

C'est le cas le plus courant : on utilise **plusieurs variables** (les colonnes de notre dataset) pour prédire $y$.

### L'Équation

L'équation s'étend pour inclure un poids pour chaque variable : $$y = w_1x_1 + w_2x_2 + \dots + w_nx_n + b$$

- **Exemple :** Prédire le prix ($y$) en fonction de la surface ($x_1$), du nombre de chambres ($x_2$), et de l'âge de la maison ($x_3$).
- **$w_1, w_2, w_n$** : Chaque variable possède son propre poids qui détermine son niveau d'importance dans la décision finale.

> [!info] Lien avec l'Algèbre Linéaire C'est ici qu'intervient la formule matricielle $Y = XW$ vue en `[[notes_alg_lin]]`. La matrice $X$ contient toutes nos données, et le vecteur $W$ contient tous nos poids.

---

## 🛠️ 4. Implémentation Pratique (Scikit-Learn)

Lien vers le script complet : `[[codepy.ipynb]]`

### A. Code : Régression Simple

Voici comment implémenter un modèle avec une seule feature.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Données (1 seule feature : Heures d'étude)
# Attention : Scikit-learn exige toujours une matrice 2D pour X
X = np.array([[1], [2], [3], [4], [5]]) 
# y : Note obtenue sur 20
y = np.array([10, 11, 14, 15, 18])

# 2. Modélisation
modele_simple = LinearRegression()
modele_simple.fit(X, y)

# 3. Paramètres et Prédiction
print(f"Poids (a) : {modele_simple.coef_[0]:.2f}")
print(f"Biais (b) : {modele_simple.intercept_:.2f}")
prediction = modele_simple.predict([[6]])
print(f"Note prédite pour 6h d'étude : {prediction[0]:.2f}")

# 4. Visualisation
plt.scatter(X, y, color='blue', label='Données réelles')
plt.plot(X, modele_simple.predict(X), color='red', label='Droite de régression')
plt.legend()
plt.show()
```

### B. Code : Régression Multiple

Ici on utilise **trois features** pour prédire le prix d'une maison : la surface ($m^2$), le nombre de chambres, et l'âge de la maison (en années).

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Données (3 features : Surface, Chambres, Âge)
# Chaque ligne = une maison
# Colonnes : [Surface (m²), Nb Chambres, Âge (ans)]
X = np.array([
    [50,  1, 30],
    [80,  2, 20],
    [100, 3, 10],
    [120, 3,  5],
    [150, 4,  2],
])
# y : Prix en milliers de dirhams (MAD)
y = np.array([400, 650, 850, 980, 1200])

# 2. Modélisation
modele_multiple = LinearRegression()
modele_multiple.fit(X, y)

# 3. Affichage des paramètres appris
features = ["Surface (m²)", "Nb Chambres", "Âge (ans)"]
print("=== Paramètres du modèle ===")
for nom, poids in zip(features, modele_multiple.coef_):
    print(f"  Poids [{nom}] : {poids:.2f}")
print(f"  Biais (b)         : {modele_multiple.intercept_:.2f}")

# 4. Prédiction sur une nouvelle maison
# Maison de 110 m², 3 chambres, 8 ans
nouvelle_maison = np.array([[110, 3, 8]])
prix_predit = modele_multiple.predict(nouvelle_maison)
print(f"\nPrix prédit pour la nouvelle maison : {prix_predit[0]:.2f} k MAD")

# 5. Évaluation du modèle sur les données d'entraînement
y_predit = modele_multiple.predict(X)
mse = mean_squared_error(y, y_predit)
print(f"MSE sur les données d'entraînement  : {mse:.2f}")
```

> [!tip] Interprétation des Poids Un poids **positif** (ex: Surface, Chambres) signifie que la feature **augmente** le prix. Un poids **négatif** (ex: Âge) signifie qu'elle le **diminue** — une maison plus ancienne vaut moins cher, toutes choses égales par ailleurs.