import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# Generate some example data (replace this with your data)
data = np.random.rand(100, 10)


# Perform PCA for dimensionality reduction
reduced_data = PCA(n_components=2).fit_transform(data)


# Determine the optimal number of clusters using the elbow method
inertia_values = []
possible_clusters = range(1, 11)  # Test for 1 to 10 clusters
for k in possible_clusters:
    kmeans = KMeans(init="k-means++", n_clusters=k, n_init=4)
    kmeans.fit(reduced_data)
    inertia_values.append(kmeans.inertia_)


# Choose the optimal number of clusters (e.g., where the curve starts to level off)
optimal_clusters = 3  # Replace with the number of clusters you determine


# Perform K-means clustering with the chosen number of clusters
kmeans = KMeans(init="k-means++", n_clusters=optimal_clusters, n_init=4)
kmeans.fit(reduced_data)


# Plot both images side by side
plt.figure(figsize=(12, 6))


# Elbow curve plot
plt.subplot(1, 2, 1)
plt.plot(possible_clusters, inertia_values, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia (Sum of Squared Distances)")


# K-means clustering result plot
plt.subplot(1, 2, 2)
h = 0.02
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


plt.imshow(Z, interpolation="nearest",
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect="auto", origin="lower")


plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker="x", s=169, linewidths=3,
            color="w", zorder=10)
plt.title("K-means clustering result (PCA-reduced data)\n"
          "Centroids are marked with white cross")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())


# Adjust layout for better visualization
plt.tight_layout()
plt.show()
