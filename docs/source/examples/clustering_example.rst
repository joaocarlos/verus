Clustering Example
==================

This example shows how to perform spatial clustering on Points of Interest.

.. code-block:: python

    from verus.clustering import GeOPTICS, KMeansHaversine
    import pandas as pd

    # Load POI data
    poi_data = pd.read_csv("../../data/poti/Porto_dataset_buffered.csv")

    # Run OPTICS clustering to obtain initial centers
    optics = GeOPTICS(
        min_samples=5,
        xi=0.05,
        min_cluster_size=5,
        verbose=True
    )
    
    optics_results = optics.run(data_source=poi_data)
    
    # Use OPTICS centers to initialize KMeans
    if optics_results["centroids"] is not None and len(optics_results["centroids"]) > 1:
        centers = optics_results["centroids"]
        print(f"Running KMeans with {len(centers)} OPTICS centers")
        
        kmeans = KMeansHaversine(
            n_clusters=len(centers),
            init="predefined",
            random_state=42,
            predefined_centers=centers
        )
        
        kmeans_results = kmeans.run(
            data_source=poi_data,
            centers_input=centers
        )
        
        # Access clustering results
        clusters = kmeans_results["clusters"]
        centroids = kmeans_results["centroids"]
        
        print(f"Found {len(centroids)} clusters")
        print(f"Cluster distribution:\n{clusters['cluster'].value_counts()}")
    
Follow this example in the project's notebooks folder.