# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist

# Load the dataset
# Replace 'wine-clustering.csv' with the path to your local dataset
df = pd.read_csv("wine-clustering.csv")

# Normalize the dataset
def normalize_data(data):
    return (data - data.min()) / (data.max() - data.min())

normalized_data = normalize_data(df.values)

# Number of clusters
num_clusters = 3

# K-means Clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans_labels = kmeans.fit_predict(normalized_data)

# K-medoids Clustering (using a simple implementation)
def k_medoids(data, k, max_iter=100):
    medoids = data[np.random.choice(len(data), k, replace=False)]
    for _ in range(max_iter):
        labels = np.argmin(cdist(data, medoids, 'euclidean'), axis=1)
        new_medoids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        if np.all(new_medoids == medoids):
            break
        medoids = new_medoids
    return labels, medoids

kmedoids_labels, kmedoids_centers = k_medoids(normalized_data, num_clusters)

# Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=num_clusters, metric='euclidean', linkage='ward')
hierarchical_labels = hierarchical.fit_predict(normalized_data)

# Plotting the clustering results
plt.figure(figsize=(18, 8))

# K-means plot
plt.subplot(1, 3, 1)
plt.scatter(normalized_data[:, 0], normalized_data[:, 1], c=kmeans_labels, cmap='viridis', marker='o', edgecolor='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# K-medoids plot
plt.subplot(1, 3, 2)
plt.scatter(normalized_data[:, 0], normalized_data[:, 1], c=kmedoids_labels, cmap='viridis', marker='o', edgecolor='k')
plt.scatter(kmedoids_centers[:, 0], kmedoids_centers[:, 1], s=200, c='blue', marker='X', label='Medoids')
plt.title('K-medoids Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Hierarchical Clustering Dendrogram
plt.subplot(1, 3, 3)
linked = linkage(normalized_data, method='ward')
dendrogram(linked, truncate_mode='lastp', p=num_clusters, show_leaf_counts=True, leaf_rotation=45., leaf_font_size=10., show_contracted=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')

plt.tight_layout()
plt.show()
