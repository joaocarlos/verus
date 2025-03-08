import time

from verus.data.timewindow import TimeWindowGenerator

# Create the generator
generator = TimeWindowGenerator(
    output_dir="../data/time_windows",
    reference_date="2025-03-07",  # First day of the week
    verbose=True,
)

# Generate time windows for the entire week
generator.generate_from_schedule(clear_existing=True)

# Create a visualization
generator.visualize_schedule(
    output_file="../data/time_windows/_time_windows_schedule.html"
)

current_time = int(time.time())
active_windows = generator.get_active_time_windows(current_time)
print(f"Active POI types at {time.ctime(current_time)}:")
for poi_type, vulnerability in active_windows.items():
    print(f"- {poi_type}: Vulnerability Index {vulnerability}")
