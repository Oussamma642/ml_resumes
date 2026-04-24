# 🔢 Algorithme ID3 — Iterative Dichotomiser 3

## 📌 Définition

**ID3** _(Iterative Dichotomiser 3)_ est un algorithme qui **construit** un arbre de décision à partir d'un dataset. Il fonctionne en apprentissage **supervisé** et résout des problèmes de **classification**.

Son principe est simple :

> À chaque étape, poser la question (choisir la variable) qui **réduit le plus le désordre** dans les données.

**Caractéristiques :**

- Fonctionne uniquement avec des **variables catégorielles** (Soleil/Pluie, Oui/Non…)
- Utilise l'**entropie** pour mesurer le désordre
- Choisit la variable avec le meilleur **gain d'information** à chaque nœud
- Construit l'arbre de façon **récursive**

---

## 🧩 Concepts clés

### 1. L'Entropie

#### L'intuition — le sac de billes 🔵🔴

Imagine un sac de billes :

|Sac|Contenu|Désordre|
|---|---|---|
|Sac A|10 billes bleues|**Nul** → tu sais déjà ce que tu vas tirer|
|Sac B|5 bleues + 5 rouges|**Maximum** → impossible de prédire|
|Sac C|8 bleues + 2 rouges|**Faible** → probablement bleue|

L'**entropie mesure exactement ce désordre**. Plus un ensemble est mélangé, plus son entropie est élevée. Plus il est pur (une seule classe), plus elle est proche de 0.

#### Formule

$$H(S) = -\sum_{k} p_k \log_2(p_k)$$

Où $p_k$ est la **proportion** d'exemples appartenant à la classe $k$ dans l'ensemble $S$.

> 💡 **Pourquoi $-\log_2$ ?** Le $\log_2$ d'une probabilité entre 0 et 1 est toujours négatif. Le signe $-$ le rend positif. Plus $p_k$ est petit (classe rare), plus $-\log_2(p_k)$ est grand — l'algorithme pénalise les surprises.

#### Cas limites

|Situation|Entropie|Signification|
|---|---|---|
|Tous les exemples = même classe|$H = 0$|Nœud **pur** — décision certaine|
|50% / 50% entre 2 classes|$H = 1$|Désordre **maximal**|
|Distribution équilibrée entre $n$ classes|$H = \log_2(n)$|Désordre absolu|

**Exemple concret :**  
Dataset de 14 jours : 9 fois "Jouer" ✅, 5 fois "Ne pas jouer" ❌

$$H(S) = -\frac{9}{14}\log_2\frac{9}{14} - \frac{5}{14}\log_2\frac{5}{14} \approx 0{,}940$$

---

### 2. Le Gain d'Information

#### L'intuition

Le gain d'information répond à cette question :

> _"Si je divise mon dataset selon la variable A, de combien est-ce que je réduis le désordre ?"_

Plus le gain est élevé, plus cette variable est **utile** pour séparer les classes → ID3 la choisit en priorité.

#### Formule

$$Gain(S, A) = H(S) - \sum_{v \in A} \frac{|S_v|}{|S|} \cdot H(S_v)$$

| Terme | Explication |  
|------|------------|  
| **S** | Ensemble de données initial (avant division) |  
| **A** | Attribut utilisé pour diviser les données |  
| **v ∈ A** | Une valeur possible de l’attribut A |  
| **Sᵥ** | Sous-ensemble de S contenant uniquement les éléments où A = v |  
| **H(S)** | Entropie de S (désordre avant division) |  
| **H(Sᵥ)** | Entropie du sous-groupe Sᵥ (désordre après division) |  
| **\|Sᵥ\| / \|S\|** | Poids du sous-groupe (sa proportion dans S) |

> 💡 C'est simplement : **Gain = Entropie avant − Entropie après (pondérée)**. On cherche la variable qui fait le plus chuter l'entropie.

---

## ⚙️ Les 6 étapes de l'algorithme

```
① Calculer H(S) — l'entropie du dataset complet
        ↓
② Pour chaque variable disponible → calculer Gain(S, A)
        ↓
③ Choisir la variable avec le Gain maximal → c'est le nœud
        ↓
④ Diviser le dataset selon les valeurs de cette variable → créer les branches
        ↓
⑤ Répéter ①②③④ récursivement sur chaque sous-groupe
        ↓
⑥ S'arrêter quand :
   • Tous les exemples d'un nœud ont la même classe  →  feuille pure ✅
   • Plus aucune variable disponible                 →  feuille par majorité
```

---

## 🎾 Exemple complet — "Jouer au Tennis ?"

### Le dataset

|Jour|Ciel|Température|Humidité|Vent|Jouer ?|
|---|---|---|---|---|---|
|J1|Soleil|Chaud|Élevée|Faible|❌ Non|
|J2|Soleil|Chaud|Élevée|Fort|❌ Non|
|J3|Couvert|Chaud|Élevée|Faible|✅ Oui|
|J4|Pluie|Doux|Élevée|Faible|✅ Oui|
|J5|Pluie|Froid|Normale|Faible|✅ Oui|
|J6|Pluie|Froid|Normale|Fort|❌ Non|
|J7|Couvert|Froid|Normale|Fort|✅ Oui|
|J8|Soleil|Doux|Élevée|Faible|❌ Non|
|J9|Soleil|Froid|Normale|Faible|✅ Oui|
|J10|Pluie|Doux|Normale|Faible|✅ Oui|
|J11|Soleil|Doux|Normale|Fort|✅ Oui|
|J12|Couvert|Doux|Élevée|Fort|✅ Oui|
|J13|Couvert|Chaud|Normale|Faible|✅ Oui|
|J14|Pluie|Doux|Élevée|Fort|❌ Non|

**Résumé :** 14 jours, 9 ✅ Oui, 5 ❌ Non

---

### 🔄 Itération N°1 — Racine de l'arbre

**Étape 1 — Entropie globale**

$$H(S) = -\frac{9}{14}\log_2\frac{9}{14} - \frac{5}{14}\log_2\frac{5}{14} \approx 0{,}940$$

---

**Étape 2 — Calcul manuel du gain pour chaque variable**

#### 🌤️ Variable : Ciel (3 valeurs)

On répartit les 14 jours selon la valeur de Ciel :

|Valeur|Jours|Composition|Entropie|
|---|---|---|---|
|Soleil|J1,J2,J8,J9,J11|[2+, 3−]|$H = -\frac{2}{5}\log_2\frac{2}{5} - \frac{3}{5}\log_2\frac{3}{5} \approx 0{,}971$|
|Couvert|J3,J7,J12,J13|[4+, 0−]|$H = 0$ (nœud pur ✅)|
|Pluie|J4,J5,J6,J10,J14|[3+, 2−]|$H = -\frac{3}{5}\log_2\frac{3}{5} - \frac{2}{5}\log_2\frac{2}{5} \approx 0{,}971$|

$$Gain(S, \text{Ciel}) = 0{,}940 - \frac{5}{14}(0{,}971) - \frac{4}{14}(0) - \frac{5}{14}(0{,}971) = \boxed{0{,}246}$$

---

#### 🌡️ Variable : Température (3 valeurs)

|Valeur|Jours|Composition|Entropie|
|---|---|---|---|
|Chaud|J1,J2,J3,J13|[2+, 2−]|$H = -\frac{2}{4}\log_2\frac{2}{4} - \frac{2}{4}\log_2\frac{2}{4} = 1{,}000$|
|Doux|J4,J8,J10,J11,J12,J14|[4+, 2−]|$H = -\frac{4}{6}\log_2\frac{4}{6} - \frac{2}{6}\log_2\frac{2}{6} \approx 0{,}918$|
|Froid|J5,J6,J7,J9|[3+, 1−]|$H = -\frac{3}{4}\log_2\frac{3}{4} - \frac{1}{4}\log_2\frac{1}{4} \approx 0{,}811$|

$$Gain(S, \text{Température}) = 0{,}940 - \frac{4}{14}(1{,}000) - \frac{6}{14}(0{,}918) - \frac{4}{14}(0{,}811) = \boxed{0{,}029}$$

---

#### 💧 Variable : Humidité (2 valeurs)

|Valeur|Jours|Composition|Entropie|
|---|---|---|---|
|Élevée|J1,J2,J3,J4,J8,J12,J14|[3+, 4−]|$H = -\frac{3}{7}\log_2\frac{3}{7} - \frac{4}{7}\log_2\frac{4}{7} \approx 0{,}985$|
|Normale|J5,J6,J7,J9,J10,J11,J13|[6+, 1−]|$H = -\frac{6}{7}\log_2\frac{6}{7} - \frac{1}{7}\log_2\frac{1}{7} \approx 0{,}592$|

$$Gain(S, \text{Humidité}) = 0{,}940 - \frac{7}{14}(0{,}985) - \frac{7}{14}(0{,}592) = \boxed{0{,}151}$$

---

#### 💨 Variable : Vent (2 valeurs)

|Valeur|Jours|Composition|Entropie|
|---|---|---|---|
|Faible|J1,J3,J4,J5,J8,J9,J10,J13|[6+, 2−]|$H = -\frac{6}{8}\log_2\frac{6}{8} - \frac{2}{8}\log_2\frac{2}{8} \approx 0{,}811$|
|Fort|J2,J6,J7,J11,J12,J14|[3+, 3−]|$H = -\frac{3}{6}\log_2\frac{3}{6} - \frac{3}{6}\log_2\frac{3}{6} = 1{,}000$|

$$Gain(S, \text{Vent}) = 0{,}940 - \frac{8}{14}(0{,}811) - \frac{6}{14}(1{,}000) = \boxed{0{,}048}$$

---

**Étape 3 — Comparaison et décision**

|Variable|Gain d'information||
|---|---|---|
|🌤️ Ciel|**0,246**|🏆 **Meilleur → racine de l'arbre**|
|💧 Humidité|0,151||
|💨 Vent|0,048||
|🌡️ Température|0,029||

> ✅ **Ciel** est choisi comme nœud racine car il réduit le plus l'entropie (0,246).

**Résultat de la division selon Ciel :**

```
            [9+, 5−]  E = 0,940
                 |
              [Ciel ?]
           /     |      \
       Soleil  Couvert  Pluie
      [2+,3−] [4+,0−]  [3+,2−]
      E=0,971   E=0    E=0,971
```

> 💡 La branche **Couvert** a une entropie de 0 → c'est une **feuille pure** ✅ Oui. Elle est terminée.  
> Les branches **Soleil** et **Pluie** ont encore du désordre → on continue récursivement.

---

### 🔄 Itération N°2a — Cas Ciel = Soleil

Sous-dataset : J1, J2, J8, J9, J11 → **[2+, 3−]**

$$H(S_\text{Soleil}) = -\frac{2}{5}\log_2\frac{2}{5} - \frac{3}{5}\log_2\frac{3}{5} \approx 0{,}971$$

Calcul des gains sur les 3 variables restantes :

|Variable|Sous-groupes|Gain|
|---|---|---|
|Température|Chaud[0+,2−] E=0 / Doux[1+,1−] E=1 / Froid[1+,0−] E=0|**0,571** 🏆|
|Humidité|Élevée[0+,3−] E=0 / Normale[2+,0−] E=0|**0,971** 🏆|
|Vent|Faible[2+,1−] E≈0,918 / Fort[1+,1−] E=1|≈0,020|

> En réalité : **Gain(S, Humidité) = 0,971** est le maximum → **Humidité** est choisie.  
> Les deux sous-groupes (Élevée et Normale) ont tous les deux E=0 → ce sont deux **feuilles pures**.

```
        [Ciel = Soleil]
              |
         [Humidité ?]
          /          \
       Élevée       Normale
      [0+,3−]       [2+,0−]
      ❌ Non         ✅ Oui
```

---

### 🔄 Itération N°2b — Cas Ciel = Pluie

Sous-dataset : J4, J5, J6, J10, J14 → **[3+, 2−]**

$$H(S_\text{Pluie}) = -\frac{3}{5}\log_2\frac{3}{5} - \frac{2}{5}\log_2\frac{2}{5} \approx 0{,}971$$

|Variable|Gain|
|---|---|
|Vent|**0,971** 🏆 — Faible[3+,0−] E=0 / Fort[0+,2−] E=0|
|Humidité|≈0,020|
|Température|≈0,020|

> **Vent** est choisi → les deux feuilles sont pures.

```
        [Ciel = Pluie]
              |
           [Vent ?]
          /         \
       Faible        Fort
      [3+,0−]       [0+,2−]
      ✅ Oui         ❌ Non
```

---

### 🌳 Arbre final construit par ID3

```
                    [Ciel ?]
               /       |        \
          Soleil     Couvert    Pluie
             |          |          |
        [Humidité?]   ✅ Oui    [Vent ?]
         /      \               /      \
      Élevée  Normale        Faible    Fort
      ❌ Non  ✅ Oui         ✅ Oui    ❌ Non
```

---

## ✅ Avantages et ⚠️ Limites

| ✅ Simple à comprendre et implémenter                                                                                       |
| -------------------------------------------------------------------------------------------------------------------------- |
| ✅ Produit des arbres **interprétables** (lisibles par un humain)                                                           |
| ✅ Efficace sur des **variables catégorielles**                                                                             |
| ⚠️ **Ne gère pas** les variables numériques continues directement                                                          |
| ⚠️ **Biais vers les variables avec beaucoup de valeurs** (une variable "Identifiant" aurait un gain énorme mais sans sens) |
| ⚠️ **Pas d'élagage (pruning)** → risque de sur-apprentissage (mémorisation du dataset)                                     |

> Ces trois limites seront corrigées par C4.5 → voir `[[3-algo-c4.5]]`

---

## 📝 Résumé en une phrase

> ID3 construit un arbre récursivement en choisissant à chaque nœud la variable **catégorielle** qui maximise le **gain d'information** (= réduit le plus l'entropie).

---

_Tags : #machine-learning #arbres-de-décision #id3 #entropie #gain-information #classification_