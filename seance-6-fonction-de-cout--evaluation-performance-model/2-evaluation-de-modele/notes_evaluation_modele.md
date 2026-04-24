# Performance des Modèles — Métriques d'Évaluation

**Dossier :** `[[performance-des-modeles]]` **Parent :** `[[seance-evaluation-du-modele]]` **Mots-clés :** #machine-learning #métriques #évaluation #régression #classification #MSE #F1-score

---

## 🎯 1. Pourquoi des Métriques ?

Les **métriques** sont des outils mathématiques qui permettent de :

- **Quantifier** l'erreur ou la justesse des prédictions
- **Comparer** différents modèles entre eux
- **Choisir** le meilleur algorithme pour un problème donné

> [!info] Régression ou Classification ? Le choix des métriques dépend du **type de problème** :
> 
> |Type de problème|Sortie|Métriques adaptées|
> |---|---|---|
> |**Régression**|Valeur numérique continue|MSE, RMSE, MAE, R²|
> |**Classification**|Catégorie / étiquette|Accuracy, Précision, Rappel, F1-score, Matrice de confusion|

---

## 📐 2. Métriques de Régression

### A. MSE — Mean Squared Error (Erreur Quadratique Moyenne)

$$\text{MSE} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$

- ✅ Pénalise fortement les **grandes erreurs** (grâce au carré)
- ❌ Sensible aux **valeurs aberrantes** (outliers)

```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, y_pred)
# MSE = 0.375
```

---

### B. MAE — Mean Absolute Error (Erreur Absolue Moyenne)

$$\text{MAE} = \frac{1}{m} \sum_{i=1}^{m} \left| \hat{y}^{(i)} - y^{(i)} \right|$$

- ✅ Plus **robuste** que la MSE, moins sensible aux outliers
- ❌ Non dérivable partout → **optimisation plus difficile**

```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)
# MAE = 0.5
```

---

### C. RMSE — Root Mean Squared Error

$$\text{RMSE} = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2}$$

- ✅ Exprimée dans la **même unité** que la variable cible → plus interprétable
- ❌ Tout comme la MSE, **sensible aux outliers**

```python
import numpy as np
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
# RMSE = 0.612
```

---

### D. R² — Coefficient de Détermination

$$R^2 = 1 - \frac{\sum_{i=1}^{m}(y^{(i)} - \hat{y}^{(i)})^2}{\sum_{i=1}^{m}(y^{(i)} - \bar{y})^2}$$

- ✅ Mesure la **proportion de variance expliquée** par le modèle
- Un $R^2 = 1$ → modèle parfait | $R^2 = 0$ → modèle aussi mauvais que la moyenne
- ❌ Peut être **trompeur** en cas de données non linéaires ou d'overfitting

```python
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)
# R² = 0.9486
```

---

### Pourquoi les métriques donnent des résultats différents ?

Toutes ces métriques sont calculées sur le **même jeu de données**, mais elles mesurent l'erreur différemment :

|Métrique|Ce qu'elle mesure|Sensibilité aux grandes erreurs|
|---|---|---|
|**MAE**|Erreur moyenne (valeur absolue)|Faible|
|**MSE**|Moyenne des erreurs au carré|Forte|
|**RMSE**|Erreur typique (racine du MSE)|Forte|
|**R²**|Part de variance expliquée|Synthèse globale|

> [!example] Exemple comparatif — Modèle A vs Modèle B
> 
> |Exemple|Vraie valeur|Modèle A|Modèle B|
> |---|---|---|---|
> |1|10|9|10|
> |2|10|9|10|
> |3|10|9|**20** ❌ (erreur énorme)|
> 
> **Modèle A :** MAE = 1.0 / MSE = 1.0 / RMSE = 1.0 **Modèle B :** MAE = 3.33 / MSE = 33.33 / RMSE = 5.77
> 
> **Conclusion :** Même si le Modèle B est parfait sur 2 exemples, **une seule grosse erreur ruine toute sa performance**. Le Modèle A est clairement plus performant car il est plus régulier et génère moins d'erreurs.

---

## 🏷️ 3. Métriques de Classification

### A. Accuracy (Exactitude)

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

- **Proportion de prédictions correctes** parmi toutes les prédictions
- ✅ Simple et intuitive
- ❌ Trompeuse si les **classes sont déséquilibrées** (ex : 95% d'une classe)

---

### B. Précision (Precision)

$$\text{Précision} = \frac{TP}{TP + FP}$$

- Proportion de **vrais positifs** parmi tous les exemples **prédits comme positifs**
- ✅ Important quand on veut **éviter les faux positifs**
- Exemple : prédire à tort qu'un patient a une maladie grave → problème ❌

---

### C. Rappel (Recall / Sensibilité)

$$\text{Rappel} = \frac{TP}{TP + FN}$$

- Proportion de **vrais positifs détectés** parmi tous les **réels positifs**
- ✅ Important quand on veut **éviter de rater des vrais cas**
- Exemple : ne pas détecter un patient réellement malade → dangereux ❌

---

### D. F1-Score

$$\text{F1-score} = 2 \cdot \frac{\text{Précision} \times \text{Rappel}}{\text{Précision} + \text{Rappel}}$$

- **Moyenne harmonique** entre Précision et Rappel
- ✅ Équilibre les deux métriques en un seul score
- ✅ Idéal quand les données sont **déséquilibrées** et qu'on veut un bon compromis

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc  = accuracy_score(y_true, y_pred)   # 0.6
prec = precision_score(y_true, y_pred)  # 0.4
rec  = recall_score(y_true, y_pred)     # 0.67
f1   = f1_score(y_true, y_pred)         # 0.5
```

---

### E. La Matrice de Confusion

Un tableau à double entrée qui compare les **prédictions** aux **valeurs réelles** et permet de visualiser précisément où le modèle se trompe.

||**Prédit : Positif**|**Prédit : Négatif**|
|---|---|---|
|**Réel : Positif**|✅ **TP** — Vrai Positif|❌ **FN** — Faux Négatif|
|**Réel : Négatif**|❌ **FP** — Faux Positif|✅ **TN** — Vrai Négatif|

- **TP** : bien classé comme positif
- **TN** : bien classé comme négatif
- **FP** : négatif mal classé comme positif _(Erreur de type I)_
- **FN** : positif non détecté _(Erreur de type II)_

> [!tip] Lien avec les métriques Toutes les métriques de classification (Accuracy, Précision, Rappel) se calculent directement depuis les 4 cases de la matrice de confusion. C'est l'outil central de l'évaluation en classification. Voir aussi : `[[regression-logistique]]` section Matrice de confusion.

---

### F. Accuracy vs Précision — L'Analogie des Fléchettes

|Cible|Accuracy|Précision|Signification|
|---|---|---|---|
|**A**|✅|✅|Excellent modèle — précis et exact|
|**B**|❌|✅|Modèle **régulier mais biaisé**|
|**C**|❌|❌|Modèle **instable et peu fiable**|
|**D**|✅|❌|Moyenne correcte mais **forte dispersion**|

> [!info] Résumé de l'analogie
> 
> - **Accuracy** = les fléchettes tombent **en moyenne** au bon endroit
> - **Précision** = les fléchettes tombent **groupées** au même endroit
> - Le modèle idéal est **A** : à la fois précis ET exact.

---

## 📋 4. Tableau Récapitulatif Global

|Métrique|Type|Formule clé|Quand l'utiliser|
|---|---|---|---|
|**MSE**|Régression|$\frac{1}{m}\sum(\hat{y}-y)^2$|Pénaliser les grosses erreurs|
|**RMSE**|Régression|$\sqrt{\text{MSE}}$|Résultat dans la même unité que $y$|
|**MAE**|Régression|$\frac{1}{m}\sum\|\hat{y}-y\|$|Données avec outliers|
|**R²**|Régression|$1 - \frac{\text{SS_res}}{\text{SS_tot}}$|Comparer à un modèle de base|
|**Accuracy**|Classification|$\frac{TP+TN}{\text{Total}}$|Classes équilibrées|
|**Précision**|Classification|$\frac{TP}{TP+FP}$|Éviter les faux positifs|
|**Rappel**|Classification|$\frac{TP}{TP+FN}$|Éviter les faux négatifs|
|**F1-score**|Classification|$2 \cdot \frac{P \times R}{P+R}$|Classes déséquilibrées|

---

## 🏁 5. Idées Clés à Retenir

> [!cite] À retenir
> 
> - Une **fonction de coût** mesure l'erreur pendant l'entraînement
> - La **descente de gradient** ajuste les paramètres pour minimiser cette erreur
> - Les **métriques d'évaluation** permettent de juger la qualité du modèle final
> - Il est important de **comprendre l'erreur**, pas seulement de la mesurer

> [!note] Prochaine étape → `[[arbres-de-decision]]` : un modèle simple à visualiser, facile à interpréter, et qui fonctionne avec peu de traitement préalable.