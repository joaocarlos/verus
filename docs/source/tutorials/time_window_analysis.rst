Time Window Analysis Tutorial
=============================

This tutorial explores how to use time-based vulnerability analysis in VERUS.

Understanding Time Windows
--------------------------

Time windows capture how vulnerability changes throughout the day:

- Businesses open and close at different hours
- Population density fluctuates during the day
- Emergency services may have different response times

Working with Time Windows
-------------------------


Create and customize time windows:

.. code-block:: python

    from verus.data import TimeWindowGenerator
    import pandas as pd
    
    # Create generator with a specific reference date
    tw_gen = TimeWindowGenerator(
        reference_date="2023-11-06",  # A Monday
        verbose=True
    )
    
    # Generate standard time windows
    time_windows = tw_gen.generate_from_schedule()
    
    # View categories
    print(f"Generated time windows for categories: {list(time_windows.keys())}")
    
    # Examine a specific category
    if "restaurant" in time_windows:
        print("\nRestaurant vulnerability throughout the day:")
        print(time_windows["restaurant"][["hour", "vi"]])

Creating Custom Time Windows
============================

You can create custom time windows for specific scenarios:

.. code-block:: python

    # Create custom time window data
    custom_tw = {}
    
    # Define morning rush hour scenario (7-9 AM)
    custom_tw["morning_rush"] = pd.DataFrame({
        "hour": range(24),
        "vi": [0.2] * 7 + [1.0] * 2 + [0.5] * 15  # High VI from 7-9 AM
    })
    
    # Define evening rush hour scenario (5-7 PM)
    custom_tw["evening_rush"] = pd.DataFrame({
        "hour": range(24),
        "vi": [0.3] * 17 + [1.0] * 2 + [0.3] * 5  # High VI from 5-7 PM
    })
    
    # Define weekend scenario
    custom_tw["weekend"] = pd.DataFrame({
        "hour": range(24),
        "vi": [0.1] * 10 + [0.7] * 8 + [0.3] * 6  # Medium VI from 10 AM-6 PM
    })

Running Assessments with Different Time Windows
===============================================

Compare vulnerability across different time scenarios:

.. code-block:: python

    from verus import VERUS
    import pandas as pd
    
    # Assuming you have poi_data and hex_grid ready
    
    # Create empty placeholder for centroids
    empty_centroids = pd.DataFrame(columns=["latitude", "longitude"])
    
    # Initialize assessor
    assessor = VERUS(
        place_name="Porto",
        method="KM-OPTICS",
        distance_method="gaussian"
    )
    
    # Load data once
    assessor.load(
        potis_df=poi_data,
        centroids_df=empty_centroids,
        zones_gdf=hex_grid
    )
    
    # Run assessments for different time scenarios
    scenarios = {
        "Morning Rush (7-9 AM)": "morning_rush",
        "Evening Rush (5-7 PM)": "evening_rush",
        "Weekend": "weekend"
    }
    
    results = {}
    
    for scenario_name, eval_time in scenarios.items():
        print(f"\nEvaluating scenario: {scenario_name}")
        
        # Update evaluation time
        assessor.evaluation_time = eval_time
        
        # Run assessment with corresponding time window
        scenario_results = assessor.run(
            time_windows=custom_tw,
            area_boundary_path=None
        )
        
        # Store results
        results[scenario_name] = scenario_results
        
        if scenario_results["vulnerability_zones"] is not None:
            # Print summary statistics
            vz = scenario_results["vulnerability_zones"]
            print(f"Zones processed: {len(vz)}")
            print("Vulnerability statistics:")
            print(vz["VL_normalized"].describe())
            
            # Save results
            assessor.save(f"./results/{eval_time}/")

Comparing Time Window Scenarios
===============================

Visualize and compare vulnerability across scenarios:

.. code-block:: python

    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Prepare data for comparison
    comparison_data = []
    
    for scenario, res in results.items():
        if res["vulnerability_zones"] is not None:
            scenario_df = res["vulnerability_zones"][["VL_normalized"]].copy()
            scenario_df["scenario"] = scenario
            comparison_data.append(scenario_df)
    
    # Combine data
    all_scenarios = pd.concat(comparison_data)
    
    # Create boxplot comparison
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="scenario", y="VL_normalized", data=all_scenarios)
    plt.title("Vulnerability Comparison Across Time Scenarios")
    plt.ylabel("Normalized Vulnerability Level")
    plt.xlabel("Time Scenario")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

Conclusion
==========

You've now learned how to use time-based vulnerability analysis in VERUS. This approach
allows you to evaluate urban vulnerability under different temporal conditions, providing
a more comprehensive understanding of risk patterns throughout the day.