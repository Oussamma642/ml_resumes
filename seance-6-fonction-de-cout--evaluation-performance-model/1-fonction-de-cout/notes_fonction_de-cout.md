# Fonction de Coût & Descente de Gradient

**Dossier :** `[[fonction-de-cout]]` **Parent :** `[[seance-entrainement-du-model]]` **Mots-clés :** #machine-learning #fonction-de-coût #gradient #optimisation #MSE

---

## 🎯 1. Objectif Fondamental

En apprentissage supervisé, le but est d'apprendre un modèle capable de **prédire des valeurs de sortie $\hat{y}$** à partir de données d'entrée $X$, de la manière la plus précise possible.

Le modèle prédit selon la formule :

$$\hat{y} = \vec{X} \cdot \vec{w} + b$$

Pour cela, on cherche à apprendre les **bons paramètres $w$ (poids) et $b$ (biais)** qui permettent d'ajuster le modèle au plus proche des vraies sorties $y$.

> [!cite] La question centrale **Comment savoir si les prédictions sont bonnes ?**
> 
> 1. On compare les sorties **prédites** $\hat{y}$ avec les vraies sorties $y$
> 2. On mesure l'**erreur** entre les deux → c'est là qu'intervient la **fonction de coût**
> 
> _Plus l'erreur est faible, meilleur est le modèle._

---

## 📏 2. La Fonction de Coût

### Définition

> [!cite] Définition La **fonction de coût** est une formule mathématique qui mesure l'écart global entre les **prédictions du modèle** et les **valeurs réelles** sur l'ensemble des données d'apprentissage.

$$\text{Loss} = \frac{1}{2m} \sum_{i=1}^{m} \left( y^{(i)} - f(x^{(i)}) \right)^2$$

- $m$ : nombre total d'exemples dans le jeu de données
- $y^{(i)}$ : valeur réelle de l'exemple $i$
- $f(x^{(i)})$ : valeur prédite par le modèle pour l'exemple $i$

### Comment la calculer — 3 étapes

**Étape 1 — Calculer le résidu $R^{(i)}$**

Pour chaque exemple $i$, on calcule la différence entre la prédiction et la valeur réelle. On appelle cette différence le **résidu** :

$$R^{(i)} = f(x^{(i)}) - y^{(i)}$$

> [!example] Exemple concret Pour une maison dont la surface est $x^{(1)} = 95, m^2$ et le vrai prix est $y^{(1)} = 130,000, MAD$ : Si le modèle prédit $f(x^{(1)}) = 150,000, MAD$, alors : $$R^{(1)} = 150 - 130 = 20$$ Le modèle a fait une erreur de **20 000 dirhams**.

**Étape 2 — Élever les résidus au carré**

On élève chaque résidu au carré pour deux raisons : éviter que les erreurs positives et négatives s'annulent, et pénaliser davantage les grandes erreurs. On obtient la **distance euclidienne** $D^{(i)}$ :

$$D^{(i)} = \left( f(x^{(i)}) - y^{(i)} \right)^2$$

**Étape 3 — Calculer la moyenne des erreurs**

On fait la moyenne de toutes ces distances quadratiques pour obtenir la **perte globale du modèle** (la Loss) :

$$\text{Loss} = \frac{1}{2m} \sum_{i=1}^{m} \left( y^{(i)} - f(x^{(i)}) \right)^2$$

> [!info] Lien avec la MSE Cette formule est exactement l'**Erreur Quadratique Moyenne (MSE)** vue dans `[[regression-lineaire]]`, divisée par 2 pour simplifier le calcul des dérivées lors de la descente de gradient.

---

## 📉 3. Réduire l'Erreur

Pour minimiser la Loss, on ajuste les paramètres $w$ et $b$. Lorsqu'on fait varier ces valeurs, la fonction de coût change et trace des courbes caractéristiques.

- **En faisant varier $b$** → on obtient une **courbe en forme de parabole**, avec un minimum clair
- **En faisant varier $w$** → même forme de parabole, avec son propre minimum

En combinant les deux variations dans un **graphique 3D**, on obtient une **surface en forme de bol**, au fond de laquelle se trouvent les valeurs optimales $(w^_, b^_)$ qui minimisent la Loss.

> [!warning] Le problème À première vue, trouver le minimum semble facile : il suffit de regarder le point le plus bas. Mais **la machine ne voit pas cette carte complète**. Elle ne sait pas à l'avance où se situe le minimum. Elle doit apprendre à s'y diriger, **pas à pas**, en partant d'une estimation initiale aléatoire.
> 
> **Solution → La Descente de Gradient**

---

## ⛰️ 4. La Descente de Gradient

### Définition

> [!cite] Définition La **descente de gradient** est un algorithme qui calcule le gradient de la fonction coût (c'est-à-dire comment celle-ci évolue lorsque $w$ et $b$ varient légèrement), pour ensuite faire un **pas dans la direction où la fonction coût diminue**.

### L'Analogie de la Montagne

Imagine que tu es sur une montagne dans le **brouillard**. Tu ne vois rien, mais tu peux **sentir la pente sous tes pieds**. À chaque pas, tu descends dans la direction où la pente est la plus forte.

- **Le gradient** = cette pente locale
- **La descente de gradient** = l'algorithme qui te fait descendre doucement vers le minimum

### La Formule de Mise à Jour

À chaque itération, on met à jour $w$ et $b$ selon :

$$w := w - \alpha \cdot \frac{\partial J}{\partial w} \qquad b := b - \alpha \cdot \frac{\partial J}{\partial b}$$

|Symbole|Rôle|
|---|---|
|$\alpha$|**Taux d'apprentissage** (_learning rate_) — taille des pas|
|$\frac{\partial J}{\partial w}$|Dérivée partielle de la Loss par rapport à $w$ — pente locale|
|$\frac{\partial J}{\partial b}$|Dérivée partielle de la Loss par rapport à $b$ — pente locale|

> [!warning] Choisir le bon taux d'apprentissage $\alpha$
> 
> - **$\alpha$ trop petit** → descente très lente, convergence longue ⏳
> - **$\alpha$ trop grand** → risque de sauter le minimum ou de diverger 💥
> - **$\alpha$ bien choisi** → convergence rapide et stable ✅

### Les 3 Phases de la Descente de Gradient

**Itération 0 — Démarrage**

On part d'un choix **aléatoire** des paramètres $(w_0, b_0)$. On calcule le gradient (la pente) à ce point initial.

**Itération 1 — Premier ajustement**

On fait un pas dans la **direction opposée à la pente**. On obtient de nouveaux paramètres $(w_1, b_1)$. Le modèle s'améliore déjà un peu.

**Itération $n$ — Convergence**

On répète ce processus jusqu'à atteindre le **minimum**. Le modèle final correspond aux meilleurs paramètres trouvés → **erreur minimale sur les données**.

---

## 🗺️ 5. Résumé Visuel du Processus

```
Données (X, y)
      ↓
Prédiction : ŷ = Xw + b
      ↓
Calcul de la Loss : J(w, b) = 1/2m Σ(y - ŷ)²
      ↓
Calcul du gradient : ∂J/∂w  et  ∂J/∂b
      ↓
Mise à jour : w := w - α·(∂J/∂w)
              b := b - α·(∂J/∂b)
      ↓
Répéter jusqu'à convergence → Modèle optimal ✅
```

---

## 📋 6. Tableau Récapitulatif

| Concept                            | Définition                         | Formule                                                          |
| ---------------------------------- | ---------------------------------- | ---------------------------------------------------------------- |
| **Résidu** $R^{(i)}$               | Erreur sur un seul exemple         | $f(x^{(i)}) - y^{(i)}$                                           |
| **Distance euclidienne** $D^{(i)}$ | Résidu au carré (toujours positif) | $(f(x^{(i)}) - y^{(i)})^2$                                       |
| **Loss (MSE)**                     | Erreur globale du modèle           | $\frac{1}{2m}\sum(y^{(i)} - f(x^{(i)}))^2$                       |
| **Gradient**                       | Pente locale de la Loss            | $\frac{\partial J}{\partial w}$, $\frac{\partial J}{\partial b}$ |
| **Taux d'apprentissage** $\alpha$  | Taille des pas de mise à jour      | hyperparamètre à régler                                          |
| **Descente de gradient**           | Algorithme pour trouver le minimum | $w := w - \alpha \cdot \frac{\partial J}{\partial w}$            |

> [!tip] À retenir La fonction de coût **mesure** l'erreur. La descente de gradient **corrige** cette erreur, pas à pas, en ajustant $w$ et $b$ jusqu'à atteindre le minimum. C'est le mécanisme fondamental qui permet à un modèle d'**apprendre** depuis les données.