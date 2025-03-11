import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from verus.data.timewindow import TimeWindowGenerator
from verus.utils.logger import Logger

if __name__ == "__main__":
    # Create a logger for the test
    logger = Logger(name="Test_TimeWindow", verbose=True)
    logger.log(
        "Starting time window test with DataFrame-first approach...", level="info"
    )

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    logger.log(f"Using base directory: {base_dir}", level="info")

    # Example 1: Generate time windows but don't save yet
    logger.log("Example 1: Generate time windows in memory", level="info")

    # Create the generator
    generator = TimeWindowGenerator(
        reference_date="2023-11-06",  # First day of the week
        verbose=True,
    )

    # Generate time windows for the entire week (in memory)
    time_windows = generator.generate_from_schedule()

    # Print summary
    total_windows = sum(len(df) for df in time_windows.values())
    logger.log(
        f"Generated {total_windows} time windows for {len(time_windows)} POI types",
        level="success",
    )

    # Example 2: Check which POIs are currently active
    logger.log("\nExample 2: Check active time windows", level="info")
    current_time = int(time.time())
    active_windows = generator.get_active_time_windows(time_windows, current_time)
    logger.log(f"Active POI types at {time.ctime(current_time)}:", level="info")
    for poi_type, vulnerability in active_windows.items():
        logger.log(f"- {poi_type}: Vulnerability Index {vulnerability}", level="info")

    # Example 3: Save time windows and create visualization
    logger.log("\nExample 3: Save time windows to disk", level="info")
    time_windows_dir = os.path.join(base_dir, "time_windows")

    # Save the time windows
    saved_files = generator.save(time_windows, path=time_windows_dir)
    logger.log(f"Saved {len(saved_files)} time window files", level="success")

    # Create and save visualization
    viz_file = os.path.join(time_windows_dir, "time_windows_schedule.html")
    generator.visualize_schedule(time_windows, output_file=viz_file)
    logger.log(f"Saved visualization to {viz_file}", level="success")

    # Example 4: Load time windows from disk
    logger.log("\nExample 4: Load time windows from disk", level="info")

    # Create a new generator and load from files
    loaded_generator, loaded_windows = TimeWindowGenerator.from_file(
        path=time_windows_dir, verbose=True
    )

    # Verify loaded data
    total_loaded = sum(len(df) for df in loaded_windows.values())
    logger.log(
        f"Loaded {total_loaded} time windows for {len(loaded_windows)} POI types",
        level="success",
    )

    # Compare with original data
    if total_loaded == total_windows:
        logger.log(
            "Successfully verified loaded data matches original", level="success"
        )
    else:
        logger.log(
            f"Data mismatch: loaded {total_loaded}, original had {total_windows}",
            level="warning",
        )

    # Example 5: Demonstrate filtering and transformation
    logger.log("\nExample 5: Filter and transform time window data", level="info")

    # Get windows for a specific POI type
    if "school" in loaded_windows:
        school_windows = loaded_windows["school"]
        logger.log(
            f"Found {len(school_windows)} time windows for schools", level="info"
        )

        # Find high vulnerability windows (VI >= 4)
        high_vuln = school_windows[school_windows["vi"] >= 4]
        logger.log(
            f"Found {len(high_vuln)} high vulnerability time windows for schools",
            level="info",
        )

        # Example of adding a new column - duration in hours
        school_windows["duration_hours"] = (
            school_windows["te"] - school_windows["ts"]
        ) / 3600
        avg_duration = school_windows["duration_hours"].mean()
        logger.log(
            f"Average school time window duration: {avg_duration:.2f} hours",
            level="info",
        )

    logger.log("Time window tests completed successfully!", level="success")
