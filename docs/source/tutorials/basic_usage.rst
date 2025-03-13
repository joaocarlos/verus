Basic Usage Tutorial
====================

This tutorial covers the basic usage of VERUS for urban vulnerability assessment.

Setup
-----

First, import the necessary modules:

.. code-block:: python

    from verus import VERUS
    from verus.data import DataExtractor, TimeWindowGenerator
    from verus.grid import HexagonGridGenerator
    import pandas as pd
    import os

Step 1: Extract POI Data
------------------------

Extract Points of Interest (POIs) from OpenStreetMap:

.. code-block:: python

    # Define target area
    region = "Porto, Portugal"
    
    # Initialize extractor
    extractor = DataExtractor(region=region, buffer_distance=500)
    
    # Extract POIs
    poi_data = extractor.run()
    
    print(f"Extracted {len(poi_data)} POIs")

Step 2: Generate Time Windows
-----------------------------

Generate time-based vulnerability indices:

.. code-block:: python

    # Create time window generator
    tw_gen = TimeWindowGenerator(reference_date="2023-11-06")
    
    # Generate time windows
    time_windows = tw_gen.generate_from_schedule()
    
    print(f"Generated time windows for {len(time_windows)} POI categories")

Step 3: Create Analysis Grid
----------------------------

Generate a hexagonal grid for vulnerability zones:

.. code-block:: python

    # Initialize grid generator
    grid_gen = HexagonGridGenerator(region=region, edge_length=250)
    
    # Generate hexagonal grid
    hex_grid = grid_gen.run()
    
    print(f"Created hexagonal grid with {len(hex_grid)} cells")

Step 4: Perform Vulnerability Assessment
----------------------------------------

Run the complete vulnerability assessment:

.. code-block:: python

    # Initialize assessment
    assessor = VERUS(
        place_name="Porto",
        method="KM-OPTICS",
        evaluation_time="ET4",
        distance_method="gaussian",
        sigma=1000
    )
    
    # Load data
    assessor.load(
        potis_df=poi_data,
        centroids_df=pd.DataFrame(columns=["latitude", "longitude"]),
        zones_gdf=hex_grid
    )
    
    # Run assessment
    results = assessor.run(time_windows=time_windows)
    
    # Save results
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    assessor.save(output_dir)

Step 5: Visualize Results
-------------------------

Create an interactive map:

.. code-block:: python

    # Create map visualization
    map_obj = assessor.visualize()
    
    # Save map to HTML file
    map_path = os.path.join(output_dir, "vulnerability_map.html")
    map_obj.save(map_path)
    
    print(f"Interactive map saved to {map_path}")

Conclusion
----------

You've now completed a basic vulnerability assessment workflow with VERUS. 
For more advanced usage, see the Advanced Clustering and Time Window Analysis tutorials.