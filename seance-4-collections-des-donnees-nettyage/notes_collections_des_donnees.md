# Séance 4 : La Collecte de Données

**Dossier :** `[[seance-4-collections-des-donnees-nettyage]]`
**Mots-clés :** #machine-learning #dataset #scraping #données #pipeline

---

## 🎯 1. Pourquoi collecter des données ?

Avant de pouvoir entraîner un modèle de Machine Learning, il est indispensable de rassembler un ensemble d'expériences. Ces données historiques vont servir de matériel d'apprentissage pour enseigner au modèle comment accomplir sa tâche.

---

## ⚙️ 2. Le Pipeline

Voici une vue d'ensemble du processus global en Machine Learning. La collecte de données n'est que la première étape d'un flux de travail complet qui mène jusqu'à l'entraînement du modèle.

> [!tip] Aperçu visuel
> *Consulte le schéma ci-dessous pour voir comment la collecte s'intègre dans le pipeline complet :*
> [[pipeline.png]]



---

## 📡 3. Les 3 Méthodes de Collecte de Données

Selon le domaine de recherche, il existe plusieurs façons d'extraire de l'information :

### A. Collecte par Capteurs
Les données sont récupérées automatiquement et en temps réel à partir de dispositifs physiques.
* **Exemples de capteurs :** Caméras, microphones, montres connectées, capteurs de voitures autonomes.
* **Cas d'usage :** Prédire la consommation d'électricité en mesurant la tension, le courant et la température.

### B. Collecte par Formulaire (Étude / Enquête)
Les données sont collectées auprès d'individus via des questionnaires ou des enquêtes.
* **Exemples d'outils :** Google Forms, enquêtes marketing, recensements.
* **Cas d'usage :** Prédire la réussite académique d'un étudiant en lui demandant ses heures d'étude, sa motivation ou sa situation sociale.

### C. Collecte par Web Scraping
C'est l'extraction automatique de données directement depuis le code de sites web.
* **Exemples de sources :** Avis clients sur Amazon, Tweets, annonces d'emploi, sites immobiliers.
* **Cas d'usage :** Prédire le prix des voitures en aspirant la marque, l'année, le kilométrage et le prix affiché sur des annonces en ligne.

---

## 🗂️ 4. Les Types de Données

Une fois collectées, les données peuvent se présenter sous trois formes différentes :

1. **Données Structurées :** Elles sont parfaitement organisées, comme dans les bases de données relationnelles.
2. **Données Semi-structurées :** Elles ont une organisation flexible, comme les bases de données non-relationnelles, les fichiers JSON/XML ou les pages web.
3. **Données Non structurées (brutes) :** Ce sont les formats complexes comme les images, l'audio, la vidéo, les fichiers PDF ou le texte en langage naturel.

---

## 📊 5. L'Anatomie d'un Dataset

Pour le Machine Learning, on organise généralement ces informations sous forme de tableau, appelé **Dataset**.

> [!info] Structure du Tableau
> * **Les Lignes :** Chaque ligne représente une **observation** unique (une expérience, un individu, un client, etc.).
> * **Les Colonnes :** Chaque colonne représente une **variable** (une information spécifique collectée pour chaque observation).

### Variables Indépendantes vs Dépendante

Dans ce tableau, on sépare les colonnes en deux grandes catégories pour l'apprentissage supervisé :

* **Les Variables Indépendantes (*Features* ou $X$) :** Ce sont les caractéristiques qui décrivent l'individu. Elles servent à expliquer, influencer ou deviner le résultat.
* **La Variable Dépendante (*Target* ou $y$) :** C'est la variable cible, celle que le modèle cherche à **prédire** à partir des autres variables.