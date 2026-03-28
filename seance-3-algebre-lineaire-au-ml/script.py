

import numpy as np


# ====================== Etude de la similarite
# Definir des vecteurs de deux étudiants
# chaque colonne correpond à une feature, note maths, note info, nbr d'études
etd1 = np.array([85, 90,12])
etd2 = np.array([80,85, 10])

# Calcul de la similarité (produit scalaire normalisé)
similarite = np.dot(etd1, etd2) / (np.linalg.norm(etd1) * np.linalg.norm(etd2))
# Une valeur proche de 1 indique que les étudiants ont des profils similaires en termes de notes et d’heures d’étude.
print(f"La similarité entre les deux étudiants: {similarite}")

# ====================== Matrices et Opérations Matricielle
# Matrice de features
X = np.array([
    [1,2],    
    [3,4],    
    [5,6]    
])

# Vecteurs des poids
W = np.array([[0.5], [1.2]])

# Calcul de produit matriciel Y=XW
Y = X@W # np.dot(X,W)


# ====================== Déterminants et Matrices Inverses
# Definition d'une matrice A
A = np.array([[4,7], [2,6]])

# Calcul du determinant
det_A = np.linalg.det(A)
print(f"Det(A) = {det_A:.0f}")

# Vérificationd de l'inversibilité et le calcul de l'inverse
if det_A != 0:
    A_inv = np.linalg.inv(A)
    print(f"Matrice inverse: \n{A_inv}")
else:
    print("La matrice n'est pas inversible")


# ====================== Valeurs Propres et Vecteurs Propres
# Un vecteur propre représente une direction invariante sous une transformation linéaire.
# Une valeur propre quantifie l’importance de cette direction.

A = np.array([[3,1], [1,3]])

# Calcul des valeurs propres et vecteurs propres
valeurs_propres, vecteurs_propres = np.linalg.eig(A)

# ====================== Décomposition Matricielle SVD
# Defintion d'une matrice
A = np.array([
    [4,0],
    [3,-5]
])
# Decomposition SVD
U,S,Vt = np.linalg.svd(A)
# Application en ML :
    # Filtrage collaboratif (Recommendations des films, musique..)
    # Compression d'images.
    # Réduction de dimension (Latent Semantic Analysis)..
    

