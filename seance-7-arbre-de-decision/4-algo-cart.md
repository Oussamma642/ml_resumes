# 🌲 Algorithme CART — Classification And Regression Trees

## 📌 Définition

**CART** est un algorithme qui construit des arbres de décision **strictement binaires** : chaque nœud produit exactement **2 branches** (gauche et droite), jamais plus.

> 💡 C'est la différence fondamentale avec ID3 et C4.5 qui peuvent avoir 3+ branches par nœud (ex: Soleil/Couvert/Pluie). CART transforme toujours la question en **"Variable ≤ seuil ?"** — oui ou non.

**Ce qui le rend unique :**

- Fonctionne en **classification** (cible catégorielle) → mesure l'**Indice de Gini**
- Fonctionne en **régression** (cible numérique) → mesure la **MSE**
- Gère nativement les variables numériques **et** catégorielles

---

## ⚙️ Les 4 étapes communes (classification et régression)

```
① Choisir un attribut (numérique ou catégoriel)
        ↓
② Générer tous les splits possibles
   → Pour chaque attribut numérique : tester tous les seuils candidats
   → Seuil = moyenne entre deux valeurs consécutives
        ↓
③ Évaluer chaque split
   → Classification : Indice de Gini pondéré
   → Régression     : MSE pondérée
        ↓
④ Sélectionner le split qui minimise le critère d'impureté
   → Répéter récursivement sur chaque sous-groupe
```

---

## 🔵 Cas 1 — Classification (Indice de Gini)

### L'intuition

Comme l'entropie dans ID3/C4.5, le **Gini mesure le désordre** dans un groupe. Mais sa formule est plus simple et plus rapide à calculer.

> Imagine que tu tires un exemple au hasard dans un groupe, et que tu lui attribues une classe au hasard selon la distribution du groupe. Le **Gini mesure la probabilité que tu te trompes**.

- Si le groupe est pur (100% d'une classe) → tu ne peux jamais te tromper → **Gini = 0** ✅
- Si le groupe est 50/50 → tu te trompes 1 fois sur 2 → **Gini = 0,5** (maximum pour 2 classes)

### Formule

$$Gini(S) = 1 - \sum_{k=1}^{K} p_k^2$$

Où $p_k$ est la proportion d'exemples de la classe $k$ dans le groupe $S$.

> 💡 **Pourquoi $1 - \sum p_k^2$ ?** La somme $\sum p_k^2$ mesure la "concentration" des classes. Si tout est dans une classe, $\sum p_k^2 = 1$ → Gini = 0. Si tout est dispersé équitablement, $\sum p_k^2$ est faible → Gini proche de 0,5. On soustrait de 1 pour transformer une mesure de pureté en mesure d'impureté.

|Situation|Gini|Signification|
|---|---|---|
|Groupe 100% même classe|**0**|Parfaitement pur ✅|
|Groupe 50% / 50% (2 classes)|**0,5**|Désordre maximal|
|Groupe équilibré entre K classes|**$1 - \frac{1}{K}$**|Désordre absolu|

### Gini pondéré après un split

Pour comparer deux splits, on calcule le Gini pondéré des deux sous-groupes obtenus :

$$G_{pondéré} = \frac{|S_{gauche}|}{|S|} \cdot Gini(S_{gauche}) + \frac{|S_{droite}|}{|S|} \cdot Gini(S_{droite})$$

> **Objectif : minimiser ce Gini pondéré** — contrairement à l'entropie qu'on maximisait via le gain.

---

### 🎯 Exemple — "Acheter une voiture selon le salaire ?"

#### Dataset

|Salaire (DH)|Acheter ?|
|---|---|
|2 500|❌ Non|
|2 700|❌ Non|
|2 900|❌ Non|
|3 100|✅ Oui|
|3 300|✅ Oui|
|3 500|✅ Oui|
|3 700|✅ Oui|
|3 900|✅ Oui|

**Résumé :** 8 exemples, 5 ✅ Oui, 3 ❌ Non

$$Gini(S) = 1 - \left(\frac{5}{8}\right)^2 - \left(\frac{3}{8}\right)^2 = 1 - 0{,}391 - 0{,}141 = 0{,}469$$

#### Étape 2 — Génération des seuils candidats

On trie les valeurs et on calcule la moyenne entre chaque paire consécutive :

$$\frac{2500+2700}{2}=2600 \quad \frac{2700+2900}{2}=2800 \quad \frac{2900+3100}{2}=\mathbf{3000}$$ $$\frac{3100+3300}{2}=3200 \quad \frac{3300+3500}{2}=3400 \quad \frac{3500+3700}{2}=3600 \quad \frac{3700+3900}{2}=3800$$

#### Étape 3 — Évaluation de chaque seuil

|Seuil|Gauche (≤)|Gini gauche|Droite (>)|Gini droite|**Gini pondéré**|
|---|---|---|---|---|---|
|2 600|[0✅, 1❌]|0,000|[5✅, 2❌]|0,408|0,357|
|2 800|[0✅, 2❌]|0,000|[5✅, 1❌]|0,278|0,208|
|**3 000**|**[0✅, 3❌]**|**0,000**|**[5✅, 0❌]**|**0,000**|**0,000** 🏆|
|3 200|[1✅, 3❌]|0,375|[4✅, 0❌]|0,000|0,188|
|3 400|[2✅, 3❌]|0,480|[3✅, 0❌]|0,000|0,300|
|3 600|[3✅, 3❌]|0,500|[2✅, 0❌]|0,000|0,375|
|3 800|[4✅, 3❌]|0,490|[1✅, 0❌]|0,000|0,429|

> 🏆 Le seuil **3 000** est optimal avec un Gini pondéré de **0,000** — les deux groupes sont parfaitement purs !

#### Étape 4 — Arbre résultant

```
           [Salaire ≤ 3000 ?]
            /              \
          Oui              Non
    [0✅, 3❌]          [5✅, 0❌]
    ❌ Non (pur)        ✅ Oui (pur)
    Gini = 0            Gini = 0
```

> 💡 Le PDF illustre le seuil 3 400 comme exemple de calcul, mais le **vrai seuil optimal est 3 000** car il donne un Gini pondéré de 0 — une séparation parfaite entre les Non (≤3000) et les Oui (>3000).

---

## 🟠 Cas 2 — Régression (MSE)

### L'intuition

Quand la cible est un **nombre continu** (ex: prix d'une maison), on ne peut plus parler de "classes" ni de Gini. On veut que les exemples d'un même groupe aient des valeurs **le plus proches possible entre elles**.

La **MSE (Mean Squared Error)** mesure exactement ça : l'écart moyen au carré par rapport à la moyenne du groupe.

> Si tous les prix dans un groupe sont identiques → MSE = 0 → groupe parfait.  
> Si les prix sont très dispersés → MSE élevée → groupe hétérogène.

### Formule

$$MSE(S) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \bar{y})^2$$

Où $\bar{y}$ est la **moyenne** des valeurs dans le groupe $S$.

**MSE pondérée après un split :**

$$MSE_{pondérée} = \frac{|S_{gauche}|}{|S|} \cdot MSE(S_{gauche}) + \frac{|S_{droite}|}{|S|} \cdot MSE(S_{droite})$$

> **Objectif : minimiser la MSE pondérée** — le groupe le plus homogène possible.

---

### 🏠 Exemple — "Prédire le prix d'une maison selon sa surface"

#### Dataset

|Surface (m²)|Prix (kDH)|
|---|---|
|50|400|
|60|450|
|70|500|
|90|700|
|100|730|
|110|750|

#### Étape 2 — Seuils candidats

$$\frac{50+60}{2}=55 \quad \frac{60+70}{2}=65 \quad \frac{70+90}{2}=\mathbf{80} \quad \frac{90+100}{2}=95 \quad \frac{100+110}{2}=105$$

#### Étape 3 — Évaluation du seuil 80 (exemple détaillé)

**Gauche (≤ 80) :** surfaces 50, 60, 70 → prix 400, 450, 500

$$\bar{y}_{gauche} = \frac{400+450+500}{3} = 450$$

$$MSE_{gauche} = \frac{(400-450)^2 + (450-450)^2 + (500-450)^2}{3} = \frac{2500+0+2500}{3} \approx 1666{,}67$$

**Droite (> 80) :** surfaces 90, 100, 110 → prix 700, 730, 750

$$\bar{y}_{droite} = \frac{700+730+750}{3} \approx 726{,}67$$

$$MSE_{droite} = \frac{(700-726{,}67)^2 + (730-726{,}67)^2 + (750-726{,}67)^2}{3} = \frac{711+11+544}{3} \approx 422{,}22$$

$$MSE_{pondérée} = \frac{3}{6}(1666{,}67) + \frac{3}{6}(422{,}22) = \boxed{1044{,}44}$$

#### Étape 4 — Comparaison de tous les seuils

|Seuil|MSE gauche|MSE droite|**MSE pondérée**| |
|---|---|---|---|---|
|55,0|0,00|15 704,00|13 086,67||
|65,0|625,00|9 950,00|6 841,67||
|**80,0**|**1 666,67**|**422,22**|**1 044,44**|🏆 **Minimum**|
|95,0|12 968,75|100,00|8 679,17||
|105,0|17 944,00|0,00|14 953,33||

> ✅ Le seuil **80** est retenu → question : **"Surface ≤ 80 m² ?"**

#### Arbre résultant

```
         [Surface ≤ 80 m² ?]
          /               \
        Oui               Non
  [50, 60, 70 m²]    [90, 100, 110 m²]
    |   |   |         |    |     | 
400kDH 450kDH 500kDH 700kDH 730kDH 750KDH  
  Prédiction: 450 kDH  Prédiction: 726,67 kDH
  MSE = 1666,67         MSE = 422,22
```

> 💡 En régression, la **prédiction d'une feuille = la moyenne** des valeurs dans ce groupe. Ici : pour une maison ≤80m² → prix prédit = **450 kDH** ; pour >80m² → **726,67 kDH**.

---

## 📊 Entropie vs Gini — Quelle différence pratique ?

C'est une question légitime : pourquoi CART utilise Gini plutôt qu'entropie ?

|Critère|Entropie (ID3/C4.5)|Gini (CART)|
|---|---|---|
|**Formule**|$-\sum p_k \log_2 p_k$|$1 - \sum p_k^2$|
|**Calcul**|Nécessite un $\log_2$|Juste des carrés → plus rapide|
|**Sensibilité**|Pénalise plus les erreurs rares|Pénalise les erreurs fréquentes|
|**Résultat pratique**|Très similaire dans la plupart des cas||

> En pratique, les deux donnent des arbres quasi-identiques. Le Gini est préféré pour sa **rapidité de calcul**, surtout sur de grands datasets.

---

## 📊 Tableau comparatif final — ID3 vs C4.5 vs CART

|Critère|ID3|C4.5|CART|
|---|---|---|---|
|**Mesure**|Gain (Entropie)|Gain Ratio|Gini / MSE|
|**Type d'arbre**|Multi-branches|Multi-branches|**Binaire uniquement**|
|**Variables numériques**|❌|✅ Seuillage|✅ Seuillage|
|**Régression**|❌|❌|✅|
|**Élagage**|❌|✅|✅|
|**Biais multi-valeurs**|⚠️ Oui|✅ Corrigé|✅ Corrigé|

---
## 💻 Code python

```
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier, plot_tree

import matplotlib.pyplot as plt

  

# Données du tableau

data = {

    'Jour': ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9',

             'J10', 'J11', 'J12', 'J13', 'J14'],

  

    'Ciel': ['Soleil', 'Soleil', 'Couvert', 'Pluie', 'Pluie',

             'Pluie', 'Couvert', 'Soleil', 'Soleil', 'Pluie',

             'Soleil', 'Couvert', 'Couvert', 'Pluie'],

  

    'Température': ['Chaud', 'Chaud', 'Chaud', 'Doux', 'Froid',

                    'Froid', 'Froid', 'Doux', 'Froid', 'Doux',

                    'Doux', 'Doux', 'Chaud', 'Doux'],

  

    'Humidité': ['Elevée', 'Elevée', 'Elevée', 'Elevée', 'Normale',

                 'Normale', 'Normale', 'Elevée', 'Normale', 'Normale',

                 'Normale', 'Elevée', 'Normale', 'Elevée'],

  

    'Vent': ['Faible', 'Fort', 'Faible', 'Faible', 'Faible',

             'Fort', 'Fort', 'Faible', 'Faible', 'Fort',

             'Fort', 'Faible', 'Faible', 'Fort'],

  

    'Jouer': ['Non', 'Non', 'Oui', 'Oui', 'Oui',

              'Non', 'Oui', 'Non', 'Oui', 'Oui',

              'Oui', 'Oui', 'Oui', 'Non']

}

  

# Creation du Dataframe

df = pd.DataFrame(data)

  

# Ecnodage des variables categorielles

label_encoders = {}

for column in ['Ciel', 'Température', 'Humidité', 'Vent', 'Jouer']:

    le = LabelEncoder()

    df[column] = le.fit_transform(df[column])

    label_encoders[column] = le

  

# Séparation des variables explicatives et cible

X = df[['Ciel', 'Température', 'Humidité', 'Vent']]

y = df['Jouer']

  

# Creation et entrainement du modele

dtc = DecisionTreeClassifier(criterion='gini', random_state=0) # CART

dtc.fit(X,y)

  

# Affichage de l'arbre

plt.figure(figsize=(12,6))

plot_tree(dtc, feature_names=X.columns, class_names=['Non', 'Oui'], filled=True)

plt.title("Arbre CART (gini)")

plt.show()
```

## 📝 Résumé en une phrase

> CART construit un arbre **binaire** à chaque étape en cherchant le seuil qui minimise le **Gini** (classification) ou la **MSE** (régression), ce qui le rend universel et utilisable dans les deux contextes.

---

_Tags : #machine-learning #arbres-de-décision #CART #gini #MSE #classification #régression_