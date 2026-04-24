	# Séance 6 : La Régression Logistique

**Dossier :** `[[regression-logistique]]` 
**Parent :** `[[seance-6-entrainement-du-model]]` 
**Mots-clés :** #machine-learning #apprentissage-supervisé #classification #probabilité

---

## 🎯 1. Objectif Fondamental

La Régression Logistique est un algorithme d'**apprentissage supervisé**.

> [!cite] Définition 
> Son but est de prédire une **classe** (variable catégorielle) à partir d'une ou plusieurs variables explicatives (les _features_ $x$). Contrairement à la régression linéaire, elle prédit une **probabilité** comprise entre 0 et 1, qu'on convertit ensuite en classe.

> [!warning] Ne pas confondre ! 
> Malgré son nom, la Régression Logistique est un algorithme de **classification**, pas de régression. Elle prédit à quelle catégorie appartient une observation (ex : spam / non-spam, malade / sain).

Pour mesurer les erreurs du modèle, on utilise la **Log-Loss** (ou _Binary Cross-Entropy_) plutôt que la MSE :

$$\text{Log-Loss} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

- **$y_i$** : La vraie classe (0 ou 1).
- **$\hat{y}_i$** : La probabilité prédite par le modèle.

---

## 📏 2. La Régression Logistique Binaire

On l'utilise quand la variable cible n'a que **deux classes possibles** (ex : 0 ou 1, Vrai ou Faux, Chat ou Chien).

### La Fonction Sigmoïde

Le cœur de la régression logistique est la **fonction sigmoïde** $\sigma$, qui transforme n'importe quelle valeur réelle en une probabilité entre 0 et 1 :

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Avec $z = ax + b$, l'équation complète devient :

$$\hat{y} = \sigma(ax + b) = \frac{1}{1 + e^{-(ax + b)}}$$

### La Règle de Décision

Une fois la probabilité $\hat{y}$ obtenue, on applique un **seuil** (généralement 0.5) pour décider de la classe finale :

$$\text{Classe prédite} = \begin{cases} 1 & \text{si } \hat{y} \geq 0.5 \\ 0 & \text{si } \hat{y} < 0.5 \end{cases}$$

- **$a$** : Le **Poids** (_Weight_). Il détermine l'impact de $x$ sur la probabilité prédite.
- **$b$** : Le **Biais** (_Bias_). Il décale la courbe sigmoïde sur l'axe horizontal.

---

## 📐 3. La Régression Logistique Multiple

C'est le cas le plus courant : on utilise **plusieurs variables** pour prédire la classe de $y$.

### L'Équation

On calcule d'abord $z$ comme une combinaison linéaire de toutes les features, puis on applique la sigmoïde :

$$z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b$$

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

- **Exemple :** Prédire si un patient est diabétique ($y = 1$) ou non ($y = 0$) en fonction de son âge ($x_1$), son taux de glucose ($x_2$), et son IMC ($x_3$).
- **$w_1, w_2, \dots, w_n$** : Chaque variable possède son propre poids, déterminant son influence sur la probabilité finale.

> [!info] Lien avec la Régression Linéaire 
> La régression logistique est essentiellement une régression linéaire ($z = XW$, voir [[notes_alg_lin]]) à laquelle on applique la fonction sigmoïde pour "écraser" le résultat dans l'intervalle $[0, 1]$.

> [!note] Classification Multi-classes 
> Pour prédire plus de deux classes (ex : chien, chat, oiseau), on utilise la variante **Softmax** (régression logistique multinomiale), qui calcule une probabilité distincte pour chaque classe. La classe avec la probabilité la plus élevée est retenue.

---

## 🧩 4. La Matrice de Confusion

Une fois le modèle entraîné, il faut l'**évaluer** : est-ce qu'il se trompe ? Et si oui, comment ?

C'est le rôle de la **matrice de confusion** : elle dresse un bilan complet des prédictions du modèle face à la réalité.

### L'Idée Simple

Imagine un modèle qui prédit si un email est un **spam (1)** ou non **(0)**. Pour chaque email, on connaît deux choses : ce que le modèle a prédit, et la réalité. Il y a donc **4 situations possibles** :

| | Le modèle prédit **0** | Le modèle prédit **1** |
|---|---|---|
| **Réalité : 0** | ✅ **Vrai Négatif (VN)** — Correct, ce n'est pas un spam | ❌ **Faux Positif (FP)** — Erreur, ce n'est pas un spam |
| **Réalité : 1** | ❌ **Faux Négatif (FN)** — Erreur, le spam est passé inaperçu | ✅ **Vrai Positif (VP)** — Correct, le spam est bien détecté |

> [!tip] Moyen mémo-technique 
> Le **premier mot** (Vrai / Faux) dit si le modèle a **raison ou tort**. Le **second mot** (Positif / Négatif) dit ce que le modèle **a prédit**.

### Exemple Concret : Détection de Spam

Un modèle analyse **20 emails**. Voici sa matrice de confusion :

$$\begin{bmatrix} 10 & 2 \\ 1 & 7 \end{bmatrix}$$

| Résultat | Nb | Interprétation |
|---|---|---|
| ✅ Vrai Négatif (VN) | 10 | 10 emails normaux correctement identifiés |
| ❌ Faux Positif (FP) | 2 | 2 emails normaux marqués à tort comme spam |
| ❌ Faux Négatif (FN) | 1 | 1 spam qui a échappé au filtre |
| ✅ Vrai Positif (VP) | 7 | 7 spams correctement détectés et bloqués |

### L'Exactitude (Accuracy)

La métrique la plus directe : la proportion de bonnes prédictions sur le total.

$$\text{Accuracy} = \frac{VP + VN}{\text{Total}} = \frac{7 + 10}{20} = 85\%$$

> [!warning] Attention aux classes déséquilibrées ! 
> Si 95% des emails sont des non-spams, un modèle qui prédit **toujours 0** obtiendra 95% d'accuracy sans jamais détecter un seul spam. Dans ce cas, on complète l'analyse avec la **Précision**, le **Rappel** et le **Score F1**, disponibles via `classification_report` dans Scikit-Learn.

---

## 💻 Code Python : Implémentation Pratique (Scikit-Learn)

Lien vers le script complet : `[[codepy.ipynb]]`

### A. Code : Classification Binaire

Prédire si un étudiant est **admis (1) ou refusé (0)** à un examen selon ses heures de révision.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Données (1 seule feature : Heures de révision)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
# y : Admis (1) ou Refusé (0)
y = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])

# 2. Modélisation
modele_binaire = LogisticRegression()
modele_binaire.fit(X, y)

# 3. Paramètres et Prédiction
print(f"Poids (a) : {modele_binaire.coef_[0][0]:.2f}")
print(f"Biais (b) : {modele_binaire.intercept_[0]:.2f}")

prob = modele_binaire.predict_proba([[5.5]])
print(f"\nProbabilité d'être refusé  (classe 0) : {prob[0][0]:.2%}")
print(f"Probabilité d'être admis   (classe 1) : {prob[0][1]:.2%}")
print(f"Décision finale : {'Admis ✅' if modele_binaire.predict([[5.5]])[0] == 1 else 'Refusé ❌'}")

# 4. Évaluation
y_predit = modele_binaire.predict(X)
print(f"\nPrécision (Accuracy) : {accuracy_score(y, y_predit):.2%}")
print(f"Matrice de confusion :\n{confusion_matrix(y, y_predit)}")

# 5. Visualisation de la courbe sigmoïde
X_range = np.linspace(0, 11, 300).reshape(-1, 1)
probabilites = modele_binaire.predict_proba(X_range)[:, 1]

plt.scatter(X, y, color='blue', zorder=5, label='Données réelles')
plt.plot(X_range, probabilites, color='red', label='Courbe Sigmoïde')
plt.axhline(0.5, color='gray', linestyle='--', label='Seuil = 0.5')
plt.xlabel("Heures de révision")
plt.ylabel("P(Admis)")
plt.title("Régression Logistique Binaire")
plt.legend()
plt.show()