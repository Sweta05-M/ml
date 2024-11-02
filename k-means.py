import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from scipy.spatial.distance import cdist

# Load Iris dataset
iris = load_iris()
data = iris.data

# Step 1: K-Means Initial Clusters
kmeans = KMeans(n_clusters=3, init='random', max_iter=1, n_init=1, random_state=0)  # Max iter=1 for initial clusters
initial_clusters_kmeans = kmeans.fit_predict(data)
initial_centroids_kmeans = kmeans.cluster_centers_

# Display Initial Clusters
print("K-Means Initial Clusters:", initial_clusters_kmeans)
print("K-Means Initial Centroids:", initial_centroids_kmeans)

# Step 2: K-Means Final Clusters with Epoch Size
kmeans_final = KMeans(n_clusters=3, init='random', max_iter=100, n_init=1, random_state=0)
final_clusters_kmeans = kmeans_final.fit_predict(data)
epoch_size_kmeans = kmeans_final.n_iter_
final_centroids_kmeans = kmeans_final.cluster_centers_

# Calculate Error Rate for K-Means
error_rate_kmeans = -kmeans_final.score(data)  # Sum of squared distances

# Display final results for K-Means
print("\nK-Means Final Clusters:", final_clusters_kmeans)
print("K-Means Epoch Size:", epoch_size_kmeans)
print("K-Means Final Centroids:", final_centroids_kmeans)
print("K-Means Error Rate:", error_rate_kmeans)

# Optional: Plot K-Means Final Clusters
plt.scatter(data[:, 0], data[:, 1], c=final_clusters_kmeans, cmap='viridis')
plt.scatter(final_centroids_kmeans[:, 0], final_centroids_kmeans[:, 1], s=300, c='red', label='Centroids')
plt.title("K-Means Clustering on Iris Dataset")
plt.legend()
plt.show()
