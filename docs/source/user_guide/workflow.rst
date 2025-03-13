VERUS Workflow
==============

This guide explains the complete workflow for performing vulnerability assessment with VERUS.

Overview
--------

The urban vulnerability assessment workflow consists of these major steps:

1. **Extract urban data**: Obtain points of interest (POIs) from OpenStreetMap
2. **Define time scenarios**: Associate vulnerability indices based on time windows
3. **Generate analysis grid**: Create hexagonal zones for vulnerability analysis
4. **Perform spatial clustering**: Group similar POIs using OPTICS and KMeans
5. **Calculate vulnerability**: Apply distance-based metrics to compute vulnerability
6. **Apply smoothing**: Enhance spatial continuity across cluster boundaries
7. **Visualize and export**: Create maps and export results for further analysis

Example Workflow
----------------

.. code-block:: python

    # 1. Import required modules
    from verus import VERUS
    from verus.data import DataExtractor, TimeWindowGenerator
    from verus.grid import HexagonGridGenerator
    import pandas as pd

    # 2. Extract urban data
    extractor = DataExtractor(region="Porto, Portugal")
    poi_data = extractor.run()

    # 3. Define time scenarios
    tw_generator = TimeWindowGenerator(reference_date="2023-11-06")
    time_windows = tw_generator.generate_from_schedule()

    # 4. Generate analysis grid
    grid_generator = HexagonGridGenerator(region="Porto, Portugal", edge_length=250)
    hex_grid = grid_generator.run()

    # 5-7. Complete vulnerability assessment
    assessor = VERUS(
        place_name="Porto",
        method="KM-OPTICS",
        evaluation_time="ET4",
        distance_method="gaussian"
    )

    # Load data
    assessor.load(
        potis_df=poi_data,
        centroids_df=pd.DataFrame(columns=["latitude", "longitude"]),
        zones_gdf=hex_grid
    )

    # Run assessment
    results = assessor.run(time_windows=time_windows)

    # Export results
    assessor.save("./results/")

    # Create interactive map
    map_obj = assessor.visualize()

Key Parameters
--------------

Distance Methods
~~~~~~~~~~~~~~~~

VERUS supports multiple methods for calculating vulnerability based on distance:

- ``gaussian``: Gaussian weighted vulnerability (default)
- ``inverse_weighted``: Inversely weighted distance

Smoothing
~~~~~~~~~

The smoothing step improves spatial continuity by reducing sharp transitions 
between neighboring zones from different clusters:

.. code-block:: python

    # Adjust smoothing with the influence threshold parameter
    assessor.smooth_vulnerability(influence_threshold=0.3)