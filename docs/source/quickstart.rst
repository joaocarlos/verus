Quickstart
==========

This guide will help you get started with VERUS quickly.

Installation
------------

Install VERUS using pip:

.. code-block:: bash

    pip install verus

Basic Usage
-----------

Here's a minimal example that demonstrates the core functionality:

.. code-block:: python

    from verus import VERUS
    from verus.data import DataExtractor, TimeWindowGenerator
    from verus.grid import HexagonGridGenerator
    import pandas as pd
    
    # 1. Extract POI data
    extractor = DataExtractor(region="Porto, Portugal")
    poi_data = extractor.run()
    
    # 2. Generate time windows
    tw_gen = TimeWindowGenerator()
    time_windows = tw_gen.generate_from_schedule()
    
    # 3. Create hexagonal grid
    grid_gen = HexagonGridGenerator(region="Porto, Portugal", edge_length=250)
    hex_grid = grid_gen.run()
    
    # 4. Initialize vulnerability assessor
    assessor = VERUS(
        place_name="Porto",
        method="KM-OPTICS",
        evaluation_time="ET4",
        distance_method="gaussian"
    )
    
    # 5. Load data
    assessor.load(
        potis_df=poi_data,
        centroids_df=pd.DataFrame(columns=["latitude", "longitude"]),
        zones_gdf=hex_grid
    )
    
    # 6. Run assessment
    results = assessor.run(time_windows=time_windows)
    
    # 7. Save results
    assessor.save("./results/")

Next Steps
----------

- Explore detailed [examples](examples/index)
- Check out the [tutorials](tutorials/index)
- See the [API Reference](api/index) for complete documentation