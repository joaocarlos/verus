Extract POI Example
===================

This example demonstrates how to extract Points of Interest (POIs) from OpenStreetMap.

.. code-block:: python

    import os
    from verus.data import DataExtractor

    # Define target region
    region = "Porto, Portugal"
    buffer_distance = 500  # meters

    # Initialize extractor
    extractor = DataExtractor(
        region=region,
        buffer_distance=buffer_distance,
        verbose=True
    )

    # Extract POIs
    poi_data = extractor.run()

    # Inspect results
    print(f"Extracted {len(poi_data)} POIs")
    print(poi_data.head())

    # Optional: Save results
    output_dir = "./data/poti/"
    os.makedirs(output_dir, exist_ok=True)
    poi_data.to_csv(os.path.join(output_dir, "porto_poti.csv"), index=False)

Follow this example in the project's notebooks folder.