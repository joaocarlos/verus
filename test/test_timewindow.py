import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from verus.data.timewindow import TimeWindowGenerator

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    print(f"Using base directory: {base_dir}")

    # Create the generator
    generator = TimeWindowGenerator(
        output_dir=base_dir,
        reference_date="2023-11-06",  # First day of the week
        verbose=True,
    )

    # Generate time windows for the entire week
    generator.generate_from_schedule(clear_existing=True)

    # Create a visualization
    generator.visualize_schedule(output_file="_time_windows_schedule.html")

    current_time = int(time.time())
    active_windows = generator.get_active_time_windows(current_time)
    print(f"Active POI types at {time.ctime(current_time)}:")
    for poi_type, vulnerability in active_windows.items():
        print(f"- {poi_type}: Vulnerability Index {vulnerability}")
