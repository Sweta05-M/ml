import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_iris
from scipy.spatial.distance import cdist

# Load Iris dataset
iris = load_iris()
data = iris.data

# Step 1: Agglomerative Initial Clusters (one per data point)
agg_clustering_initial = AgglomerativeClustering(n_clusters=None, distance_threshold=0)
initial_clusters_agg = agg_clustering_initial.fit_predict(data)
print("\nAgglomerative Initial Clusters (one per data point):", initial_clusters_agg)

# Step 2: Agglomerative Final Clusters
agg_clustering_final = AgglomerativeClustering(n_clusters=3)
final_clusters_agg = agg_clustering_final.fit_predict(data)

# Optional: Plot Agglomerative Final Clusters
plt.scatter(data[:, 0], data[:, 1], c=final_clusters_agg, cmap='viridis')
plt.title("Agglomerative Clustering - Final Clusters on Iris Dataset")
plt.show()

# Dendrogram for visualizing hierarchical steps
plt.figure(figsize=(10, 7))
plt.title("Dendrogram for Iris Dataset")
linkage_matrix = linkage(data, method='ward')
dendrogram(linkage_matrix)
plt.show()

# Step 3: Calculate Error Rate for Agglomerative Clustering
# Calculate centroids based on final clusters
centroids_agg = []
for i in range(3):  # Assuming 3 clusters
    cluster_points = data[final_clusters_agg == i]
    centroid = cluster_points.mean(axis=0)
    centroids_agg.append(centroid)

# Calculate error rate as sum of squared distances to centroids
error_rate_agg = sum(np.min(cdist(data, centroids_agg, 'euclidean')**2, axis=1))

# Display final results for Agglomerative Clustering
print("\nAgglomerative Final Clusters:", final_clusters_agg)
print("Agglomerative Error Rate:", error_rate_agg)
print("Total linkage steps (proxy for epochs):", len(linkage_matrix))
