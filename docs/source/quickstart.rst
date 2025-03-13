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
    config = {
        "max_vulnerability": {"Porto": 0.0003476593149558199},
    }
    assessor = VERUS(place_name="Porto", config=config)
    
    # 5. Load data
    assessor.load(
        potis_df=poi_data,
        time_windows_dict=time_windows,
        zones_gdf=hex_grid
    )
    
    # 6. Run assessment
    evaluation_time = tw_gen.to_unix_epoch("2025-03-13 17:30:00")
    results = assessor.run(evaluation_time=evaluation_time)
    
    # 7. Save results
    assessor.save("./results/")
    assessor.visualize("./results/Porto_2025-03-13_17:30:00")

Follow this example in the project's notebooks folder.

Next Steps
----------

- Understand the :doc:`core concepts <concepts/index>`
- Explore detailed :doc:`examples <examples/index>`
- Check out the :doc:`tutorials <tutorials/index>`
- See the :doc:`API Reference <api/index>` for complete documentation