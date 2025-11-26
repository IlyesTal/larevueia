import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# --- 1. Chargement des données (Cas réel) ---
# Le dataset Iris contient 150 échantillons de 3 espèces de fleurs différentes.
iris = load_iris()
X = iris.data  # Les mesures (features)

# Pour la lisibilité, on le met dans un DataFrame (optionnel mais recommandé)
df = pd.DataFrame(X, columns=iris.feature_names)
print("Aperçu des données brutes :")
print(df.head())

# --- 2. Préparation (Mise à l'échelle) ---
# Comme vu dans l'article, on normalise pour que les "cm" aient tous le même poids
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. Configuration & Entraînement ---
# On choisit k=3 car on sait qu'il existe 3 espèces dans ce dataset.
# Dans la vraie vie (sans savoir), on aurait utilisé la méthode du coude.
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans.fit(X_scaled)

# --- 4. Exploitation des résultats ---
# On récupère les labels (0, 1 ou 2) attribués par l'algo
clusters = kmeans.labels_

# On ajoute le cluster au DataFrame pour analyse
df['cluster_kmeans'] = clusters

# --- 5. Visualisation (La preuve par l'image) ---
# On visualise 2 dimensions : Longueur du pétale vs Largeur du pétale
plt.figure(figsize=(10, 6))

# Affichage des points, colorés selon leur cluster K-Means
plt.scatter(X_scaled[:, 2], X_scaled[:, 3], c=clusters, cmap='viridis', s=50, alpha=0.8, label='Fleurs')

# Affichage des Centroïdes (les croix rouges)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 2], centers[:, 3], c='red', s=200, marker='X', label='Centroïdes')

plt.title('Résultat du K-Means sur le dataset Iris')
plt.xlabel('Longueur du pétale (normalisée)')
plt.ylabel('Largeur du pétale (normalisée)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Comparaison rapide (optionnelle)
print("\nRépartition des fleurs par cluster trouvé :")
print(df['cluster_kmeans'].value_counts())
