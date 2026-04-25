généralement avec c4.5, pour le nœud racine, on choisit une variable qui combine entre un bon gain d'information qui réduit le maximum possible de l'entropie et donne un sous groupe avec des résultats des classes pures, et aussi on prend en considération le piège de la diversité des valeurs de cette variable qui peut créer beaucoup de sous groupes inutiles, cela peut être résumé dans le mot clé gain ration, plus ce gain ratio est élevé, meilleur la variable est.

# 🌿 Algorithme C4.5 — L'évolution d'ID3

## 📌 Pourquoi C4.5 ? — Les limites d'ID3

Avant de définir C4.5, il faut comprendre _pourquoi_ ID3 ne suffit pas. Il souffre de **3 problèmes concrets** :

### Problème 1 — Le biais vers les variables à nombreuses valeurs

Imagine un dataset avec une colonne **"ID_Client"** : chaque client a un identifiant unique. ID3 lui attribuerait un gain d'information **maximal** (chaque sous-groupe = 1 seul exemple = entropie 0), alors que cette variable est totalement inutile pour prédire quoi que ce soit sur de nouvelles données.

> ID3 confond _"variable qui fragmente beaucoup"_ avec _"variable qui informe beaucoup"_.

### Problème 2 — Variables numériques non gérées

ID3 ne peut pas traiter directement des valeurs comme **Température = 72°** ou **Âge = 35**. Il faudrait les discrétiser manuellement à l'avance.

### Problème 3 — Sur-apprentissage (Overfitting)

Sans mécanisme d'arrêt intelligent, ID3 continue de diviser jusqu'à avoir des feuilles pures. L'arbre **mémorise** le dataset d'entraînement au lieu d'**apprendre** des règles généralisables.

---

## 🔧 Les 3 améliorations de C4.5

|Problème ID3|Solution C4.5|
|---|---|
|Biais vers variables à beaucoup de valeurs|**Gain Ratio** (gain normalisé)|
|Ne gère pas les variables numériques|**Seuillage automatique**|
|Sur-apprentissage|**Élagage (Pruning)**|

---

## 🧩 Amélioration 1 — Le Gain Ratio

### L'intuition

Le gain d'information récompense les variables qui créent _beaucoup_ de sous-groupes. La solution de C4.5 est simple :

> **Diviser le gain par la taille de la division elle-même.**  
> Une variable qui crée 10 branches minuscules sera pénalisée.  
> Une variable qui crée 2 branches riches sera valorisée.

C'est le rôle du **Split Information** : mesurer à quel point une division est large/fragmentée — indépendamment des classes.

### Formules

**Split Information** — mesure la fragmentation de la division :

$$SplitInfo(S, A) = -\sum_{i=1}^{n} \frac{|S_i|}{|S|} \cdot \log_2\left(\frac{|S_i|}{|S|}\right)$$

> 💡 C'est exactement la formule de l'entropie, mais appliquée **aux tailles des sous-groupes** (et non aux classes). Plus les branches sont nombreuses et équilibrées, plus le Split Information est élevé.

**Gain Ratio** — le gain normalisé :

$$GainRatio(S, A) = \frac{Gain(S, A)}{SplitInfo(S, A)}$$

### Exemple — Comparer Outlook vs Vent vs Humidité

On repart du même dataset Tennis (14 jours, 9✅ 5❌) vu dans `[[2-algo-id3]]`.

#### 🌤️ Outlook (3 valeurs : Sunny=5, Overcast=4, Rain=5)

$$SplitInfo(S, \text{Outlook}) = -\left(\frac{5}{14}\log_2\frac{5}{14} + \frac{4}{14}\log_2\frac{4}{14} + \frac{5}{14}\log_2\frac{5}{14}\right) \approx 1{,}577$$

$$GainRatio(S, \text{Outlook}) = \frac{0{,}246}{1{,}577} \approx \boxed{0{,}156}$$

#### 💧 Humidité (2 valeurs : Élevée=7, Normale=7)

$$SplitInfo(S, \text{Humidité}) = -\left(\frac{7}{14}\log_2\frac{7}{14} + \frac{7}{14}\log_2\frac{7}{14}\right) = 1{,}000$$

$$GainRatio(S, \text{Humidité}) = \frac{0{,}151}{1{,}000} = \boxed{0{,}151}$$

> 💡 Humidité est parfaitement équilibrée (7/7), donc son Split Info = 1 (maximum pour 2 valeurs). Son gain ratio = gain directement.

#### 💨 Vent (2 valeurs : Faible=8, Fort=6)

$$SplitInfo(S, \text{Vent}) = -\left(\frac{8}{14}\log_2\frac{8}{14} + \frac{6}{14}\log_2\frac{6}{14}\right) \approx 0{,}985$$

$$GainRatio(S, \text{Vent}) = \frac{0{,}048}{0{,}985} \approx \boxed{0{,}049}$$

#### 📊 Comparaison finale

|Variable|Gain (ID3)|Split Info|Gain Ratio (C4.5)||
|---|---|---|---|---|
|🌤️ Outlook|0,246|1,577|**0,156**|🏆 Choisi|
|💧 Humidité|0,151|1,000|0,151||
|💨 Vent|0,048|0,985|0,049||

> ✅ Le classement reste le même ici, mais ce n'est pas toujours le cas. Avec une variable "ID_Client" (14 valeurs uniques), le gain aurait été maximum mais le Split Info aussi (≈3,807), ce qui effondre le Gain Ratio à ~0.

---

## 🧩 Amélioration 2 — Gestion des variables numériques

### L'idée

Pour une variable numérique (ex: Température), C4.5 transforme automatiquement la question _"Quelle est la température ?"_ en une question binaire _"La température est-elle ≤ seuil ?"_.

Il teste **tous les seuils candidats** et choisit celui qui maximise le Gain Ratio.

### Les étapes

```
① Trier les valeurs de la variable par ordre croissant
        ↓
② Calculer tous les seuils candidats
   → Seuil = moyenne entre chaque paire de valeurs consécutives
        ↓
③ Pour chaque seuil, diviser en 2 groupes :
   → Gauche : valeurs ≤ seuil
   → Droite : valeurs > seuil
        ↓
④ Calculer Gain et Split Info de chaque division
        ↓
⑤ Garder le seuil avec le meilleur Gain Ratio
```

### Exemple — Température numérique

Dataset trié par température :

|Temp|64|65|68|69|70|71|72|72|75|75|80|81|83|85|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Jouer|✅|❌|✅|✅|✅|❌|❌|✅|✅|✅|❌|✅|✅|❌|

Les seuils candidats sont les moyennes entre valeurs consécutives : 64,5 — 66,5 — 68,5 — 69,5 — 70,5 — 71,5 — 72,0 — 73,5 — 75,0 — 77,5 — 80,5 — 82,0 — 84,0…

#### Calcul détaillé pour le seuil 84,0

C'est le seuil qui sépare **≤84** (13 exemples) et **>84** (1 exemple) :

|Groupe|Exemples|Composition|Entropie|
|---|---|---|---|
|≤ 84|J1 à J13 (sauf J14)|[9+, 4−]|$H = -\frac{9}{13}\log_2\frac{9}{13} - \frac{4}{13}\log_2\frac{4}{13} \approx 0{,}891$|
|> 84|J14 (Temp=85, Non)|[0+, 1−]|$H = 0$ (pur ❌)|

$$Gain = 0{,}940 - \frac{13}{14}(0{,}891) - \frac{1}{14}(0) \approx 0{,}113$$

$$SplitInfo = -\left(\frac{13}{14}\log_2\frac{13}{14} + \frac{1}{14}\log_2\frac{1}{14}\right) \approx 0{,}371$$

$$GainRatio = \frac{0{,}113}{0{,}371} \approx \boxed{0{,}305}$$

#### Tableau des meilleurs seuils candidats

|Seuil|Gauche (≤)|Droite (>)|Gain Ratio|
|---|---|---|---|
|**84,0**|13|1|**0,305** 🏆|
|64,5|1|13|0,128|
|70,5|5|9|0,048|
|77,5|10|4|0,029|

> ✅ Le seuil **84,0** est retenu → la question devient : **"Température ≤ 84 ?"**  
> Le nœud est binaire : branche gauche (très probable Oui) et branche droite (Non).

---

## 🧩 Amélioration 3 — L'Élagage (Pruning)

### Le problème du sur-apprentissage

Un arbre construit jusqu'au bout (comme ID3 le fait) ressemble à ça :

```
Dataset d'entraînement :   L'arbre mémorise tout → 100% de précision ✅
Nouvelles données :        L'arbre ne sait pas généraliser → mauvaise précision ❌
```

C'est l'**overfitting** : l'arbre a appris _par cœur_ au lieu d'apprendre _des règles_.

### L'analogie de la révision

> Imagine réviser un examen en mémorisant toutes les réponses du vieux partiel mot pour mot.  
> Si les questions changent légèrement → tu échoues.  
> La bonne révision, c'est comprendre les concepts → tu t'adaptes.  
> L'élagage, c'est forcer l'arbre à "comprendre" plutôt que "mémoriser".

### Comment fonctionne le Pruning

C4.5 utilise le **Pruning par erreur pessimiste** (post-pruning) :

```
① Construire l'arbre complet jusqu'aux feuilles pures
        ↓
② Remonter depuis les feuilles vers la racine
        ↓
③ Pour chaque nœud interne, poser la question :
   "Si je remplace ce sous-arbre par une feuille (la classe majoritaire),
    est-ce que le taux d'erreur estimé se dégrade significativement ?"
        ↓
④ Si non → on élagua (remplace par une feuille)
   Si oui → on garde le sous-arbre
```

### Résultat visuel

```
Avant élagage :                    Après élagage :

      [Ciel ?]                           [Ciel ?]
    /    |    \                        /    |    \
Sunny  Over  Rain               Sunny  Over  Rain
  |      |     |                  |      |     |
[Hum?] ✅Oui [Vent?]           [Hum?] ✅Oui  ✅Oui ← feuille simplifiée
 / \         /   \              / \
❌  ✅      ✅    ❌            ❌  ✅
```

> La branche Rain avait peut-être des erreurs marginales liées au bruit du dataset → on la simplifie en feuille ✅Oui car c'est la classe majoritaire dans ce sous-groupe.

---

## 📊 Tableau comparatif — ID3 vs C4.5

|Critère|ID3|C4.5|
|---|---|---|
|**Mesure de choix**|Gain d'information|Gain Ratio|
|**Variables numériques**|❌ Non supporté|✅ Seuillage automatique|
|**Variables catégorielles**|✅|✅|
|**Élagage**|❌ Aucun|✅ Pruning pessimiste|
|**Biais multi-valeurs**|❌ Oui (favorise)|✅ Corrigé|
|**Risque d'overfitting**|⚠️ Élevé|✅ Réduit|
|**Complexité**|Simple|Modérée|

---

## 📝 Résumé en une phrase

> C4.5 améliore ID3 sur trois fronts : il normalise le gain avec le **Gain Ratio** pour éviter le biais, gère automatiquement les **variables numériques** par seuillage, et réduit l'overfitting grâce à l'**élagage**.

---

_Tags : #machine-learning #arbres-de-décision #c4.5 #gain-ratio #pruning #variables-numériques #classification_