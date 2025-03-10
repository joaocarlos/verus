import csv
import os
import shutil
from datetime import datetime, timedelta

import pandas as pd

from verus.utils.logger import Logger
from verus.utils.paths import PathManager


class TimeWindowGenerator(Logger):
    """
    Class for generating temporal influence time windows.

    This class provides methods for creating and managing time windows
    that define when certain POI types are active and their vulnerability indices.
    """

    DEFAULT_SCHEDULES = {
        "hospital": [
            {"days": "Weekdays", "start": "07:00", "end": "10:00", "vulnerability": 5},
            {"days": "Weekdays", "start": "10:00", "end": "16:00", "vulnerability": 2},
            {"days": "Weekdays", "start": "16:00", "end": "19:00", "vulnerability": 5},
            {"days": "Weekends", "start": "00:00", "end": "23:59", "vulnerability": 1},
        ],
        "park": [
            {"days": "Weekdays", "start": "16:00", "end": "20:00", "vulnerability": 2},
            {"days": "Weekends", "start": "08:00", "end": "18:00", "vulnerability": 3},
        ],
        "mall": [
            {"days": "Weekdays", "start": "12:00", "end": "14:00", "vulnerability": 3},
            {"days": "Weekdays", "start": "17:00", "end": "20:00", "vulnerability": 4},
            {"days": "Weekends", "start": "09:00", "end": "20:00", "vulnerability": 2},
        ],
        "school": [
            {"days": "Weekdays", "start": "08:00", "end": "10:00", "vulnerability": 4},
            {"days": "Weekdays", "start": "16:00", "end": "18:00", "vulnerability": 4},
        ],
        "attraction": [
            {"days": "Weekdays", "start": "09:00", "end": "17:00", "vulnerability": 3},
            {"days": "Weekends", "start": "09:00", "end": "17:00", "vulnerability": 5},
        ],
        "metro_station": [
            {"days": "Weekdays", "start": "07:00", "end": "09:00", "vulnerability": 5},
            {"days": "Weekdays", "start": "12:00", "end": "14:00", "vulnerability": 3},
            {"days": "Weekdays", "start": "17:00", "end": "19:00", "vulnerability": 5},
        ],
        "train_station": [
            {"days": "Weekdays", "start": "07:00", "end": "09:00", "vulnerability": 5},
            {"days": "Weekdays", "start": "12:00", "end": "14:00", "vulnerability": 3},
            {"days": "Weekdays", "start": "17:00", "end": "19:00", "vulnerability": 5},
        ],
        "bus_station": [
            {"days": "Weekdays", "start": "07:00", "end": "09:00", "vulnerability": 5},
            {"days": "Weekdays", "start": "12:00", "end": "14:00", "vulnerability": 3},
            {"days": "Weekdays", "start": "17:00", "end": "19:00", "vulnerability": 5},
        ],
        "university": [
            {"days": "Weekdays", "start": "07:00", "end": "09:00", "vulnerability": 4},
            {"days": "Weekdays", "start": "17:00", "end": "19:00", "vulnerability": 4},
        ],
        "industrial": [
            {"days": "Weekdays", "start": "08:00", "end": "17:00", "vulnerability": 3},
            {"days": "Weekends", "start": "00:00", "end": "00:00", "vulnerability": 0},
        ],
    }

    def __init__(
        self,
        output_dir=None,
        reference_date=None,
        schedules=None,
        verbose=True,
    ):
        """
        Initialize the TimeWindowGenerator.

        Args:
            output_dir (str): Directory to store time window files
            reference_date (datetime or str, optional): Reference date for week generation
                                                       Default is Monday of current week
            schedules (dict, optional): Custom POI schedules. If None, uses default
            verbose (bool): Whether to print log messages
        """
        super().__init__(verbose=verbose)

        # Initialize path manager
        self.paths = PathManager(output_dir=output_dir)
        self.time_windows_dir = self.paths.get_path("time_windows")

        # Set reference date (defaults to most recent Monday)
        if reference_date is None:
            today = datetime.now()
            # Get most recent Monday (0 = Monday in datetime.weekday())
            days_since_monday = today.weekday()
            self.reference_date = today - timedelta(days=days_since_monday)
        elif isinstance(reference_date, str):
            self.reference_date = datetime.strptime(reference_date, "%Y-%m-%d")
        else:
            self.reference_date = reference_date

        self.log(f"Using reference date: {self.reference_date.strftime('%Y-%m-%d')}")

        # Set schedules
        self.schedules = schedules if schedules else self.DEFAULT_SCHEDULES
        self.log(f"Loaded schedules for {len(self.schedules)} POI types")

    @staticmethod
    def to_unix_epoch(date_time_str):
        """
        Convert datetime string to UNIX epoch timestamp.

        Args:
            date_time_str (str): Datetime string format 'YYYY-MM-DD HH:MM:SS'

        Returns:
            int: UNIX timestamp (seconds since epoch)
        """
        dt_object = datetime.strptime(date_time_str, "%Y-%m-%d %H:%M:%S")
        return int((dt_object - datetime(1970, 1, 1)).total_seconds())

    @staticmethod
    def from_unix_epoch(timestamp):
        """
        Convert UNIX timestamp to datetime string.

        Args:
            timestamp (int): UNIX epoch timestamp

        Returns:
            str: Formatted datetime string
        """
        dt_object = datetime(1970, 1, 1) + timedelta(seconds=timestamp)
        return dt_object.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def is_weekend(date_str):
        """
        Check if the given date is a weekend.

        Args:
            date_str (str): Date string in format 'YYYY-MM-DD'

        Returns:
            bool: True if weekend, False otherwise
        """
        day = pd.to_datetime(date_str)
        return day.weekday() > 4  # 0 is Monday, 6 is Sunday

    def add_days(self, days):
        """
        Add days to the reference date.

        Args:
            days (int): Number of days to add

        Returns:
            str: Resulting date string in format 'YYYY-MM-DD'
        """
        result_date = self.reference_date + timedelta(days=days)
        return result_date.strftime("%Y-%m-%d")

    def create_time_window(
        self, poti_type, vulnerability, start_time, end_time, replace=False
    ):
        """
        Create a single time window entry.

        Args:
            poti_type (str): POI type identifier
            vulnerability (int): Vulnerability index (0-5)
            start_time (str): Start time in format 'YYYY-MM-DD HH:MM:SS'
            end_time (str): End time in format 'YYYY-MM-DD HH:MM:SS'
            replace (bool): Whether to replace existing file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate inputs
            if not 0 <= vulnerability <= 5:
                self.log(
                    f"Invalid vulnerability value: {vulnerability}. Must be 0-5.",
                    "error",
                )
                return False

            file_path = os.path.join(self.time_windows_dir, f"{poti_type}.csv")
            file_exists = os.path.exists(file_path)

            # Determine write mode
            mode = "w" if replace or not file_exists else "a"

            # Convert times to timestamps
            start_timestamp = self.to_unix_epoch(start_time)
            end_timestamp = self.to_unix_epoch(end_time)

            # Write to file
            with open(file_path, mode, newline="") as csvfile:
                fieldnames = ["vi", "ts", "te"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if not file_exists or replace:
                    writer.writeheader()

                writer.writerow(
                    {"vi": vulnerability, "ts": start_timestamp, "te": end_timestamp}
                )

            self.log(
                f"Added time window for {poti_type}: {start_time} to {end_time} (VI: {vulnerability})"
            )
            return True

        except Exception as e:
            self.log(f"Error creating time window: {str(e)}", "error")
            return False

    def generate_from_schedule(self, clear_existing=False):
        """
        Generate time windows from predefined schedules.

        Args:
            clear_existing (bool): Whether to clear the output directory first

        Returns:
            int: Number of time windows created
        """
        # Clear existing files if requested
        if clear_existing and os.path.exists(self.time_windows_dir):
            self.log(f"Clearing existing time windows in {self.time_windows_dir}")
            shutil.rmtree(self.time_windows_dir)
            os.makedirs(self.time_windows_dir)

        # Count created windows
        windows_created = 0

        # Generate time windows for each POI type
        for poti_type, schedules in self.schedules.items():
            self.log(f"Processing schedule for {poti_type}")

            for schedule in schedules:
                # Determine which days to add based on weekday/weekend
                if schedule["days"] == "Weekdays":
                    days_to_add = range(0, 5)  # Monday (0) to Friday (4)
                else:  # Weekends
                    days_to_add = range(5, 7)  # Saturday (5) to Sunday (6)

                for day in days_to_add:
                    # Get the current date
                    current_date = self.add_days(day)

                    # Skip if day type doesn't match (shouldn't happen with proper ranges)
                    if (
                        schedule["days"] == "Weekdays" and self.is_weekend(current_date)
                    ) or (
                        schedule["days"] == "Weekends"
                        and not self.is_weekend(current_date)
                    ):
                        continue

                    # Format start and end times
                    start_time = f"{current_date} {schedule['start']}:00"
                    if schedule["end"] != "23:59":
                        end_time = f"{current_date} {schedule['end']}:59"
                    else:
                        end_time = f"{current_date} {schedule['end']}:00"

                    # Create the time window
                    success = self.create_time_window(
                        poti_type=poti_type,
                        vulnerability=schedule["vulnerability"],
                        start_time=start_time,
                        end_time=end_time,
                        replace=False,  # Append to existing file
                    )

                    if success:
                        windows_created += 1

        self.log(f"Successfully created {windows_created} time windows")
        return windows_created

    def get_active_time_windows(self, timestamp=None):
        """
        Get all active time windows for a specific timestamp.

        Args:
            timestamp (int, optional): UNIX timestamp to check. Default is current time.

        Returns:
            dict: Dictionary mapping POI types to their vulnerability indices
        """
        if timestamp is None:
            timestamp = int(datetime.now().timestamp())

        active_windows = {}

        try:
            for file in os.listdir(self.time_windows_dir):
                if file.endswith(".csv"):
                    poi_type = os.path.splitext(file)[0]
                    file_path = os.path.join(self.time_windows_dir, file)

                    # Read the CSV
                    df = pd.read_csv(file_path)

                    # Find active windows
                    active = df[(df["ts"] <= timestamp) & (df["te"] >= timestamp)]

                    if not active.empty:
                        # Use the maximum vulnerability if multiple windows are active
                        active_windows[poi_type] = active["vi"].max()

            return active_windows

        except Exception as e:
            self.log(f"Error retrieving active time windows: {str(e)}", "error")
            return {}

    def visualize_schedule(self, output_file="time_windows_schedule.html"):
        """
        Create an HTML visualization of the time window schedule.

        Args:
            output_file (str): Output HTML file path

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import plotly.figure_factory as ff
            import plotly.io as pio

            # Collect all time windows
            all_windows = []

            # Process each POI type
            for poti_type, schedules in self.schedules.items():
                for schedule in schedules:
                    # Create a task for each day type
                    if schedule["days"] == "Weekdays":
                        day_names = [
                            "Monday",
                            "Tuesday",
                            "Wednesday",
                            "Thursday",
                            "Friday",
                        ]
                    else:
                        day_names = ["Saturday", "Sunday"]

                    for day in day_names:
                        # Format for Gantt chart
                        all_windows.append(
                            {
                                "Task": f"{poti_type} ({day})",
                                "Start": f"2023-01-01 {schedule['start']}:00",
                                "Finish": f"2023-01-01 {schedule['end']}:00",
                                "Resource": f"VI: {schedule['vulnerability']}",
                                "Description": f"{poti_type} - {day} - VI: {schedule['vulnerability']}",
                            }
                        )

            # Create DataFrame
            df = pd.DataFrame(all_windows)

            # Convert to datetime
            df["Start"] = pd.to_datetime(df["Start"])
            df["Finish"] = pd.to_datetime(df["Finish"])

            # Create the Gantt chart
            fig = ff.create_gantt(
                df,
                colors={f"VI: {i}": self._get_vi_color(i) for i in range(6)},
                index_col="Resource",
                show_colorbar=True,
                group_tasks=True,
                title="POI Time Windows Schedule",
            )

            # Write to HTML file
            pio.write_html(fig, os.path.join(self.time_windows_dir, output_file))
            self.log(f"Schedule visualization saved to {output_file}")
            return True

        except Exception as e:
            self.log(f"Error creating visualization: {str(e)}", "error")
            return False

    def _get_vi_color(self, vulnerability):
        """Get color based on vulnerability index"""
        colors = [
            "rgb(240,240,240)",  # 0 - Grey
            "rgb(191,255,191)",  # 1 - Light green
            "rgb(152,251,152)",  # 2 - Pale green
            "rgb(255,240,98)",  # 3 - Yellow
            "rgb(255,167,87)",  # 4 - Orange
            "rgb(255,105,97)",  # 5 - Red
        ]
        return colors[min(vulnerability, 5)]

    def clear_time_windows(self):
        """
        Clear all time window files.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if os.path.exists(self.time_windows_dir):
                self.log(f"Clearing all time windows in {self.time_windows_dir}")

                for file in os.listdir(self.time_windows_dir):
                    if file.endswith(".csv"):
                        os.remove(os.path.join(self.time_windows_dir, file))

                self.log("Time windows cleared successfully")
                return True
            return False
        except Exception as e:
            self.log(f"Error clearing time windows: {str(e)}", "error")
            return False
