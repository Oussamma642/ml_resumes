# Algèbre Linéaire pour le Machine Learning

**Dossier :** `[[algebre-lineaire]]` **Parent :** `[[notes_alg_lin]]` **Mots-clés :** #machine-learning #algèbre-linéaire #matrices #vecteurs #PCA #SVD

---

## 🎯 1. Pourquoi l'Algèbre Linéaire ?

L'algèbre linéaire est le **langage mathématique du Machine Learning**. Quand on travaille avec des données, on doit les représenter sous une forme mathématique — c'est exactement son rôle.

> [!cite] Définition L'algèbre linéaire permet de **représenter**, **manipuler** et **transformer** des données pour produire des prédictions. Elle est utilisée dans **tous** les modèles de ML et permet de faire des calculs efficaces sur de grandes quantités de données.

Elle sert à trois choses fondamentales :

- **Représenter les données** → sous forme de vecteurs et de matrices
- **Manipuler les données** → opérations mathématiques sur ces structures
- **Faire des prédictions** → via le produit matriciel $Y = XW$

---

## 📐 2. Les Vecteurs

### Définition

Un **vecteur** est une liste ordonnée de nombres. En Machine Learning, il représente une **observation** (un individu, un exemple).

$$v = \begin{bmatrix} v_1 \ v_2 \ v_3 \ \vdots \ v_n \end{bmatrix}$$

> [!info] En ML : une personne = un vecteur Chaque valeur du vecteur correspond à une **feature** (caractéristique) de l'observation.

### Exemples concrets

**Exemple 1 — Profil académique d'un étudiant :**

$$s = \begin{bmatrix} 85 \ 90 \ 12 \end{bmatrix} \quad \begin{array}{l} \leftarrow \text{Note en Maths (/ 100)} \ \leftarrow \text{Note en Informatique (/ 100)} \ \leftarrow \text{Heures d'étude / semaine} \end{array}$$

**Exemple 2 — Profil d'un utilisateur e-commerce :**

$$u = \begin{bmatrix} 30 \ 5 \ 120 \end{bmatrix} \quad \begin{array}{l} \leftarrow \text{Âge (ans)} \ \leftarrow \text{Nombre d'achats récents} \ \leftarrow \text{Temps sur l'app (min/jour)} \end{array}$$

> [!tip] Application : Similarité entre vecteurs On peut mesurer à quel point deux observations se ressemblent grâce au **produit scalaire normalisé** (cosine similarity). Une valeur proche de **1** signifie que les profils sont très similaires.
> 
> ```python
> import numpy as np
> etudiant1 = np.array([85, 90, 12])
> etudiant2 = np.array([80, 85, 10])
> similarite = np.dot(etudiant1, etudiant2) / (np.linalg.norm(etudiant1) * np.linalg.norm(etudiant2))
> print(f"Similarité : {similarite:.4f}")  # → 0.9999
> ```

---

## 🗂️ 3. Les Matrices

### Définition

Une **matrice** est un ensemble de vecteurs organisé en lignes et colonnes. Elle représente un **dataset entier**.

$$X = \begin{bmatrix} 20 & 3000 \ 30 & 4000 \ 22 & 2800 \end{bmatrix}$$

> [!cite] Règle fondamentale **Les lignes = individus (exemples)** **Les colonnes = caractéristiques (features)**

### Dimensions

- $m$ = nombre de lignes = nombre d'**exemples**
- $n$ = nombre de colonnes = nombre de **variables (features)**
- On dit que la matrice est de dimension $m \times n$

### Opérations importantes

- **Addition / Soustraction** : élément par élément
- **Multiplication matricielle** : essentielle pour les réseaux de neurones
- **Produit scalaire** : mesure la similarité entre vecteurs
- **Transposée** : réorganise les données (lignes ↔ colonnes)

---

## ⚡ 4. Le Produit Matriciel $Y = XW$ — Le Cœur du ML

C'est la formule la plus importante du Machine Learning. Elle décrit comment un modèle produit une prédiction à partir des données.

$$Y = XW$$

|Symbole|Signification|Dimension|
|---|---|---|
|$X$|Matrice des données (features)|$m \times n$|
|$W$|Vecteur des poids (ce que le modèle apprend)|$n \times 1$|
|$Y$|Vecteur des prédictions|$m \times 1$|

### Exemple numérique

Avec âge = 25, salaire = 3000, et les poids $w_1 = 0.1$, $w_2 = 0.001$ :

$$Y = w_1 \times \text{âge} + w_2 \times \text{salaire} = (0.1 \times 25) + (0.001 \times 3000) = 2.5 + 3 = \mathbf{5.5}$$

Quand on a beaucoup de données, ce calcul s'écrit de façon compacte : $Y = XW$

> [!info] Lien avec Scikit-Learn Quand on appelle `model.fit(X, y)`, le modèle cherche les **meilleures valeurs de $W$**. **Apprendre = trouver les bonnes valeurs de $W$.**
> 
> - `fit()` → apprentissage du modèle (trouver $W$)
> - `predict()` → faire des prédictions ($Y = XW$)

```python
import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6]])   # Matrice des features
W = np.array([[0.5], [1.2]])              # Vecteur des poids

Y = np.dot(X, W)   # Produit matriciel → Prédictions
# Y = [[2.9], [6.3], [9.7]]
```

---

## 🚧 5. Limites : Le Problème de la Dimension

En pratique, les données réelles sont bien plus complexes que des exemples à 2 variables.

|Données simples|Données réelles|
|---|---|
|âge|âge|
|salaire|salaire, expérience, niveau d'étude|
||ville, compétences, comportement...|

> [!warning] Le problème Plus il y a de variables, plus le problème devient difficile :
> 
> - Calcul plus compliqué
> - Modèle plus lent
> - Difficile à comprendre et à visualiser
> 
> **"On peut avoir 10, 50, 100 variables..."**

C'est pour résoudre ce problème qu'on utilise les **valeurs propres** et le **PCA**.

---

## 🧭 6. Valeurs Propres et Vecteurs Propres

### Intuition

Dans un nuage de points, certaines **directions** contiennent plus d'information que d'autres. L'idée est de trouver ces directions pour garder l'essentiel et ignorer le reste.

> [!cite] Définition
> 
> - Un **vecteur propre** est une direction importante dans les données — une direction qui ne change pas de sens sous une transformation linéaire.
> - Une **valeur propre** ($\lambda$) indique combien cette direction est importante.
> 
> $$Av = \lambda v$$

### Règle simple

- **Grande valeur propre** → direction très importante ✅
- **Petite valeur propre** → direction peu importante, peut être ignorée

### Application en ML

Le calcul des valeurs et vecteurs propres est la base du **PCA** — l'algorithme de réduction de dimension.

```python
import numpy as np

A = np.array([[3, 1], [1, 3]])

valeurs_propres, vecteurs_propres = np.linalg.eig(A)
print("Valeurs propres :", valeurs_propres)     # [4. 2.]
print("Vecteurs propres :\n", vecteurs_propres)
```

> [!tip] Interprétation Les **vecteurs propres ne changent pas de direction** sous la transformation — seule leur longueur est affectée par la valeur propre associée. Cette propriété est exploitée par le PCA pour détecter les axes de variance maximale.

---

## 🔬 7. PCA — Analyse en Composantes Principales

### Problème résolu

Trop de variables → données complexes, modèle lent → on veut **réduire le nombre de variables** sans perdre l'information essentielle.

### Définition

> [!cite] Définition Le **PCA** (_Principal Component Analysis_) est une technique qui simplifie les données en **gardant les directions où l'information est maximale** et en éliminant le bruit.
> 
> - **Vecteurs propres** → les nouvelles directions (axes)
> - **Valeurs propres** → l'importance de chaque direction
> - **PCA** → utilise ces directions pour projeter les données

> [!warning] Le PCA ne supprime pas les données ! Il les **transforme** pour garder l'essentiel. On garde les directions à grande valeur propre, on élimine celles à faible valeur propre.

### Cas d'usage

- Réduction de dimension pour des **images**
- Traitement de **données médicales**
- **Reconnaissance faciale** : PCA extrait les caractéristiques principales (yeux, bouche, nez) sous forme de vecteurs propres

---

## 🔩 8. Déterminants et Matrices Inverses

### Le Déterminant

Un nombre scalaire qui indique si une matrice est **inversible**.

$$\det(A) \neq 0 \Rightarrow \text{Matrice inversible (pas de dépendance linéaire)}$$ $$\det(A) = 0 \Rightarrow \text{Matrice singulière (problème de multicolinéarité)}$$

### La Matrice Inverse

$A^{-1}$ est l'inverse de $A$ si :

$$A \cdot A^{-1} = I$$

où $I$ est la **matrice identité**.

### Application en ML — Régression Linéaire

La formule des **moindres carrés** utilise l'inversion matricielle pour trouver les poids optimaux $W$ :

$$W = (X^T X)^{-1} X^T Y$$

```python
import numpy as np

A = np.array([[4, 7], [2, 6]])

det_A = np.linalg.det(A)
print(f"Déterminant : {det_A:.2f}")   # → 10.0

if det_A != 0:
    A_inv = np.linalg.inv(A)
    print("Matrice inverse :\n", A_inv)
else:
    print("La matrice A n'est pas inversible.")
```

---

## 🧩 9. Décomposition Matricielle (SVD)

### La SVD — Singular Value Decomposition

La SVD décompose n'importe quelle matrice $A$ en trois matrices :

$$A = U \Sigma V^T$$

|Matrice|Rôle|
|---|---|
|$U$|Vecteurs propres normalisés des **lignes** de $A$|
|$\Sigma$|Matrice diagonale — les **valeurs singulières** $\sigma_i = \sqrt{\lambda_i}$|
|$V^T$|Vecteurs propres normalisés des **colonnes** de $A$|

> [!info] Lien avec les valeurs propres Les valeurs singulières $\sigma_i$ sont les racines carrées des valeurs propres des matrices $AA^T$ et $A^TA$, triées par ordre décroissant.

### Applications en ML

- **Compression d'images** (format JPEG)
- **Réduction de dimension** — alternative au PCA
- **Systèmes de recommandation** (Netflix, Spotify) — filtrage collaboratif

```python
import numpy as np

A = np.array([[4, 0], [3, -5]])

U, S, Vt = np.linalg.svd(A)
print("Matrice U:\n", U)
print("Valeurs singulières:", S)   # [6.32, 3.16]
print("Matrice V^T:\n", Vt)
```

---

## 📋 10. Tableau Récapitulatif

|Concept|Définition|Utilisation en ML|
|---|---|---|
|**Vecteur**|Liste de nombres = une observation|Représenter un individu / des features|
|**Matrice**|Ensemble de vecteurs = un dataset|Stocker toutes les données|
|**Produit matriciel** $Y=XW$|Combinaison données × poids|Produire des prédictions|
|**Déterminant**|Scalaire d'inversibilité|Vérifier si un système est soluble|
|**Matrice inverse**|$AA^{-1} = I$|Résoudre $W = (X^TX)^{-1}X^TY$|
|**Valeurs/Vecteurs propres**|Directions importantes des données|Base du PCA|
|**PCA**|Réduction de dimension|Simplifier les données complexes|
|**SVD**|Décomposition $A = U\Sigma V^T$|Compression, recommandation|

---

## 🏁 Conclusion

> [!cite] À retenir **Toute l'algèbre linéaire sert à transformer des données en prédictions.**
> 
> - Vecteurs → données
> - Matrices → dataset
> - Produit matriciel → prédiction
> 
> _"On ne fait pas des maths pour les maths... on fait des maths pour comprendre les modèles."_