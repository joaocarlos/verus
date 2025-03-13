Advanced Clustering Tutorial
============================

This tutorial explains the advanced clustering techniques used in VERUS.

Understanding Clustering in VERUS
----------------------------------

VERUS uses a hybrid clustering approach that combines:

1. **OPTICS** - For detecting density-based clusters and finding initial centers
2. **KMeans with Haversine distance** - For refinement with geographic distance awareness

Customizing OPTICS Parameters
-----------------------------

The OPTICS algorithm has several key parameters that can be tuned:

.. code-block:: python

    from verus.clustering import GeOPTICS
    
    # Create custom OPTICS instance
    optics = GeOPTICS(
        min_samples=10,        # Minimum samples in a neighborhood
        xi=0.05,               # Steepness threshold
        min_cluster_size=8,    # Minimum cluster size
        max_eps=1000,          # Maximum neighborhood radius (meters)
        verbose=True
    )
    
    # Run clustering
    optics_results = optics.run(data_source=poi_data)
    
    # Access results
    clusters = optics_results["clusters"]
    centroids = optics_results["centroids"]
    
    # Visualizing OPTICS results
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 8))
    plt.scatter(
        clusters["longitude"], 
        clusters["latitude"], 
        c=clusters["cluster"], 
        cmap="viridis", 
        alpha=0.6
    )
    plt.scatter(
        centroids["longitude"], 
        centroids["latitude"], 
        c="red", 
        marker="x", 
        s=100
    )
    plt.title("OPTICS Clustering Results")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

Customizing KMeans Parameters
-----------------------------

KMeans can be further customized:

.. code-block:: python

    from verus.clustering import KMeansHaversine
    
    # Create custom KMeans instance
    kmeans = KMeansHaversine(
        n_clusters=10,         # Number of clusters
        init="k-means++",      # Initialization method
        random_state=42,       # For reproducibility
        max_iter=300,          # Maximum iterations
        verbose=True
    )
    
    # Run KMeans
    kmeans_results = kmeans.run(data_source=poi_data)
    
    # Access results
    clusters = kmeans_results["clusters"]
    centroids = kmeans_results["centroids"]

Hybrid Clustering Pipeline
--------------------------

For best results, combine OPTICS and KMeans:

.. code-block:: python

    # 1. Run OPTICS to get initial centers
    optics = GeOPTICS(min_samples=5, xi=0.05, min_cluster_size=5)
    optics_results = optics.run(data_source=poi_data)
    
    # 2. Use these centers to initialize KMeans
    centers = optics_results["centroids"]
    
    kmeans = KMeansHaversine(
        n_clusters=len(centers),
        init="predefined",
        predefined_centers=centers,
        random_state=42
    )
    
    # 3. Run KMeans with OPTICS centers
    kmeans_results = kmeans.run(
        data_source=poi_data,
        centers_input=centers
    )
    
    # 4. Access final results
    final_clusters = kmeans_results["clusters"]
    final_centroids = kmeans_results["centroids"]

Evaluating Clustering Quality
-----------------------------

To evaluate clustering quality:

.. code-block:: python

    from sklearn import metrics
    
    # Calculate silhouette score (requires scikit-learn)
    # First, extract coordinates and convert to radians for Haversine distance
    import numpy as np
    from haversine import haversine
    
    coords = final_clusters[["latitude", "longitude"]].values
    labels = final_clusters["cluster"].values
    
    # Define custom distance matrix
    def create_distance_matrix(coords):
        n = len(coords)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = haversine(
                    (coords[i][0], coords[i][1]), 
                    (coords[j][0], coords[j][1])
                )
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        return distance_matrix
    
    # Calculate distance matrix
    distances = create_distance_matrix(coords)
    
    # Calculate silhouette score
    silhouette = metrics.silhouette_score(
        distances, 
        labels, 
        metric="precomputed"
    )
    
    print(f"Silhouette score: {silhouette}")

Conclusion
----------

You've now learned how to customize and optimize the clustering process in VERUS. 
This knowledge can help you achieve better vulnerability assessments by creating 
more meaningful spatial clusters.