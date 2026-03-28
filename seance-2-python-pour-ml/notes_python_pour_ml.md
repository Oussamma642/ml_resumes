# Séance 2 : L'écosystème Python pour le Machine Learning

**Mots-clés :** #python #numpy #pandas #matplotlib #scikit-learn #machine-learning

Python est le langage de référence en Data Science et Machine Learning grâce à sa syntaxe simple, son orientation objet et surtout son écosystème extrêmement riche de bibliothèques spécialisées.

---

## 🧮 1. NumPy (Numerical Python)

NumPy est la bibliothèque fondamentale pour le calcul scientifique et la manipulation de données numériques. Écrite en C et Fortran, elle est beaucoup plus rapide que les listes Python classiques.

### Structures de données (Tableaux / ndarrays)
* **Vecteur (1D) :** Liste de nombres (ex: série temporelle).
* **Matrice (2D) :** Tableau organisé en lignes et colonnes, base de l'algèbre linéaire.
* **Tenseurs (3D, 4D+) :** Utilisés en Deep Learning (ex: traitement d'images).

### Création et Manipulation
* **Fonctions de génération :** `np.zeros()` (rempli de 0), `np.ones()` (rempli de 1), `np.random.rand()` (valeurs aléatoires).
* **Attributs :** `.shape` donne les dimensions de la matrice, `.size` donne le nombre total d'éléments.
* **Transformation :** `.reshape()` permet de réorganiser les dimensions d'un tableau sans changer ses données.

### Opérations Mathématiques
* **Élément par élément :** Addition (`+`) ou produit d'Hadamard (`*`).
* **Produit Matriciel :** S'effectue avec la fonction `np.dot(A, B)`.

> [!warning] Limites de NumPy
> NumPy gère très mal les données hétérogènes (le texte), les valeurs manquantes (NaN), et ne permet pas de nommer les colonnes.

---

## 🐼 2. Pandas (Panel Data)

Pandas vient combler les lacunes de NumPy. C'est l'outil ultime pour la manipulation, le nettoyage et l'analyse de données tabulaires hétérogènes sous forme de **DataFrames**.

### Importation et Exploration
* **Chargement :** Permet de lire facilement des fichiers externes avec `pd.read_csv()`.
* **Aperçu :** `.head()` (premières lignes), `.tail()` (dernières lignes).
* **Analyse globale :** `.info()` (structure et types de données), `.describe()` (statistiques générales : moyenne, min, max...).

### Manipulation et Transformation
* **Colonnes :** Ajout de nouvelles colonnes (`df['Nouvelle'] = ...`), sélection (`df['Nom']`) ou suppression (`df.drop(columns=[...])`).
* **Filtrage :** Permet d'isoler des données selon des conditions (ex: `df[df['Âge'] > 30]`).
* **Agrégation :** Regroupement des données avec `.groupby()` et tri avec `.sort_values()`.

> [!warning] Limites de Pandas
> Les capacités de visualisation graphique de Pandas sont basiques et manquent d'interactivité pour explorer des distributions complexes.

---

## 📊 3. Matplotlib

Matplotlib (via son module `pyplot`) est la bibliothèque standard pour créer des visualisations de données statiques afin d'évaluer les résultats des modèles.

### Les Types de Graphiques
* **Courbe (`plt.plot`) :** Évolution d'une variable continue. Possibilité de superposer plusieurs courbes (ex: Sinus, Cosinus).
* **Histogramme (`plt.hist`) :** Montre la distribution et la fréquence d'une variable numérique.
* **Graphique en barres (`plt.bar` / `plt.barh`) :** Idéal pour comparer différentes catégories.
* **Nuage de points (`plt.scatter`) :** Visualise la relation (corrélation) entre deux variables (ex: Âge vs Revenu).
* **Diagramme circulaire (`plt.pie`) :** Montre la répartition en pourcentages d'un ensemble.
* **Boîte à moustaches (`plt.boxplot`) :** Indispensable pour voir la distribution des quartiles et détecter les **valeurs aberrantes (outliers)** situées hors des moustaches.
* **Graphique en aire (`plt.fill_between`) :** Montre l'évolution d'une tendance dans le temps avec une zone remplie.

> [!warning] Limites de Matplotlib
> Ne fait "que" dessiner. Il ne contient aucun algorithme de Machine Learning et ne permet pas le prétraitement des données.

---

## 🤖 4. Scikit-Learn

Scikit-learn est la bibliothèque de modélisation par excellence. Elle rassemble les algorithmes de Machine Learning prêts à être entraînés.

### Fonctionnalités principales
Elle couvre tout le cycle de vie d'un projet ML :
1. **Prétraitement des données**.
2. **Apprentissage (Training)** à partir des données historiques.
3. **Prédiction** sur de nouvelles données.
4. **Évaluation** et sélection des meilleurs modèles.

### Algorithmes et Datasets
* **Domaines couverts :** Algorithmes de Régression, de Classification, et de Clustering.
* **Datasets intégrés :** Contient des jeux de données d'entraînement (ex: `load_iris()`) très utiles pour tester ses modèles rapidement, en séparant les caractéristiques ($X$) et les cibles à prédire ($y$).