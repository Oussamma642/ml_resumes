# 🌳 Arbres de Décision — Définition Générale

## 🧠 Intuition avant tout

Avant de définir quoi que ce soit, imagine cette situation :

> Tu veux décider si tu vas sortir faire du sport dehors aujourd'hui.  
> Tu te poses des questions dans l'ordre :
> 
> - _Est-ce qu'il pleut ?_ → Oui → ❌ Je reste à la maison
> - Non → _Est-ce qu'il fait trop chaud (> 35°) ?_ → Oui → ❌ Trop risqué
> - Non → _Est-ce que j'ai du temps libre ?_ → Oui → ✅ Je sors !

Tu viens de construire **un arbre de décision dans ta tête**. C'est exactement ça : une série de questions logiques, posées dans le bon ordre, qui mènent à une décision finale.

---

## 📌 Définition formelle

> Un **arbre de décision** est un modèle prédictif **arborescent** qui effectue des choix successifs sur les données pour aboutir à une **prédiction** (une classe ou une valeur).

Il apprend ces choix **automatiquement à partir des données d'entraînement**, en trouvant les questions les plus pertinentes à poser.

---

## 🏗️ Anatomie d'un arbre de décision

```
              [Ciel ?]                   ← Nœud Racine (Root)
             /    |    \
        Soleil  Nuageux  Pluie
          |        |        |
     [Humidité?]  ✅Oui   [Vent?]        ← Nœuds internes (questions)
       /    \              /    \
   Élevée  Normale     Fort   Faible
     |        |          |       |
   ❌Non    ✅Oui       ❌Non   ✅Oui    ← Feuilles (décisions finales)
```

|Composant|Rôle|
|---|---|
|**Nœud Racine** (Root Node)|Le premier test — la variable la plus informative|
|**Nœuds internes**|Chaque question posée sur une variable|
|**Branches**|Les valeurs possibles d'une variable (Soleil / Nuageux / Pluie)|
|**Feuilles** (Leaf Nodes)|La décision finale — la classe prédite|

---

## ⚠️ Modèle vs Algorithme : ne pas confondre

C'est une distinction importante que beaucoup oublient :

| [[2-algo-id3]]   |**Modèle**|**Algorithme**|
|---|---|---|
| **C'est quoi ?** |La structure arborescente finale|La méthode qui **construit** l'arbre|
| **Analogie**     |Le plan d'une maison construite|L'architecte + les règles de construction|
| **Exemples**     |L'arbre avec ses nœuds et feuilles|ID3, C4.5, CART|

> 💡 ID3, C4.5 et CART ne sont pas des arbres — ce sont des **recettes** pour en construire un à partir de données.

---

## 🎯 Cas d'usage

Les arbres de décision s'utilisent dans deux grandes familles de problèmes :

### Classification

La cible est une **catégorie** (oui/non, classe A/B/C...)

- Diagnostic médical : _Ce patient est-il malade ?_
- Filtrage spam : _Cet email est-il un spam ?_
- RH : _Ce CV est-il à retenir ?_

### Régression

La cible est une **valeur numérique continue**

- Prédiction de prix : _Combien vaut cette maison ?_
- Prévision de notes : _Quelle note va avoir cet étudiant ?_

---

## ✅ Pourquoi utiliser un arbre de décision ?

|Avantage|Explication|
|---|---|
|**Interprétable**|On peut lire et expliquer les règles facilement|
|**Pas de prétraitement**|Pas besoin de normaliser les données|
|**Polyvalent**|Fonctionne en classification ET en régression|
|**Visuel**|On peut le dessiner et le montrer à n'importe qui|

---

## 🔗 Ce qui vient ensuite

Dans les notes suivantes, on va voir comment **construire** cet arbre avec deux algorithmes :

- `[[2-algo-id3]]` → **ID3** : utilise l'entropie et le gain d'information
- `[[3-algo-c4.5]]` → **C4.5** : améliore ID3 avec le gain ratio et l'élagage

---

_Tags : #machine-learning #arbres-de-décision #classification #régression #supervisé_