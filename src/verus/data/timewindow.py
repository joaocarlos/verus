"""
Time window generation module for the Verus project.

This module provides functionality to generate and manage time windows that
define when certain POI types are active and their vulnerability indices.
"""

import os
from datetime import datetime, timedelta

import pandas as pd

from verus.utils.logger import Logger


class TimeWindowGenerator(Logger):
    """
    Generate and manage temporal influence time windows.

    This class provides methods for creating and managing time windows
    that define when certain POI types are active and their vulnerability indices.
    Time windows are represented as DataFrames and can be optionally saved to disk.

    Attributes:
        reference_date (datetime): Reference date for week generation.
        schedules (dict): Dictionary mapping POI types to their schedules.

    Examples:
        >>> generator = TimeWindowGenerator(reference_date="2023-11-06")
        >>> time_windows = generator.generate_from_schedule()
        >>> active_windows = generator.get_active_time_windows(time_windows)
        >>> generator.save(time_windows, path="./data/time_windows")
    """

    DEFAULT_SCHEDULES = {
        "hospital": [
            {
                "days": "Weekdays",
                "start": "07:00",
                "end": "10:00",
                "vulnerability": 0.8,
            },
            {
                "days": "Weekdays",
                "start": "10:00",
                "end": "16:00",
                "vulnerability": 0.4,
            },
            {
                "days": "Weekdays",
                "start": "16:00",
                "end": "19:00",
                "vulnerability": 0.8,
            },
            {
                "days": "Weekends",
                "start": "00:00",
                "end": "23:59",
                "vulnerability": 0.2,
            },
        ],
        "park": [
            {
                "days": "Weekdays",
                "start": "16:00",
                "end": "20:00",
                "vulnerability": 0.2,
            },
            {
                "days": "Weekends",
                "start": "08:00",
                "end": "18:00",
                "vulnerability": 0.4,
            },
        ],
        "mall": [
            {
                "days": "Weekdays",
                "start": "12:00",
                "end": "14:00",
                "vulnerability": 0.7,
            },
            {
                "days": "Weekdays",
                "start": "17:00",
                "end": "20:00",
                "vulnerability": 0.8,
            },
            {
                "days": "Weekends",
                "start": "09:00",
                "end": "20:00",
                "vulnerability": 0.4,
            },
        ],
        "school": [
            {
                "days": "Weekdays",
                "start": "08:00",
                "end": "10:00",
                "vulnerability": 0.7,
            },
            {
                "days": "Weekdays",
                "start": "16:00",
                "end": "18:00",
                "vulnerability": 0.7,
            },
        ],
        "attraction": [
            {
                "days": "Weekdays",
                "start": "09:00",
                "end": "17:00",
                "vulnerability": 0.6,
            },
            {
                "days": "Weekends",
                "start": "09:00",
                "end": "17:00",
                "vulnerability": 1.0,
            },
        ],
        "station": [
            {
                "days": "Weekdays",
                "start": "07:00",
                "end": "09:00",
                "vulnerability": 0.5,
            },
            {
                "days": "Weekdays",
                "start": "12:00",
                "end": "14:00",
                "vulnerability": 0.2,
            },
            {
                "days": "Weekdays",
                "start": "17:00",
                "end": "19:00",
                "vulnerability": 0.5,
            },
        ],
        "bus_station": [
            {
                "days": "Weekdays",
                "start": "07:00",
                "end": "09:00",
                "vulnerability": 0.6,
            },
            {
                "days": "Weekdays",
                "start": "12:00",
                "end": "14:00",
                "vulnerability": 0.4,
            },
            {
                "days": "Weekdays",
                "start": "17:00",
                "end": "19:00",
                "vulnerability": 0.6,
            },
        ],
        "university": [
            {
                "days": "Weekdays",
                "start": "07:00",
                "end": "09:00",
                "vulnerability": 0.7,
            },
            {
                "days": "Weekdays",
                "start": "17:00",
                "end": "19:00",
                "vulnerability": 0.7,
            },
        ],
        "industrial": [
            {
                "days": "Weekdays",
                "start": "08:00",
                "end": "17:00",
                "vulnerability": 0.3,
            },
        ],
    }

    def __init__(
        self,
        reference_date=None,
        schedules=None,
        verbose=True,
    ):
        """
        Initialize the TimeWindowGenerator.

        Parameters
        ----------
        reference_date : datetime or str, optional
            Reference date for week generation. Default is Monday of current week.
        schedules : dict, optional
            Custom POI schedules. If None, uses default.
        verbose : bool
            Whether to print log messages.
        """
        super().__init__(verbose=verbose)

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

        Parameters
        ----------
        date_time_str : str
            Datetime string format 'YYYY-MM-DD HH:MM:SS'

        Returns
        -------
        int
            UNIX timestamp (seconds since epoch)
        """
        dt_object = datetime.strptime(date_time_str, "%Y-%m-%d %H:%M:%S")
        return int((dt_object - datetime(1970, 1, 1)).total_seconds())

    @staticmethod
    def from_unix_epoch(timestamp):
        """
        Convert UNIX timestamp to datetime string.

        Parameters
        ----------
        timestamp : int
            UNIX epoch timestamp

        Returns
        -------
        str
            Formatted datetime string
        """
        dt_object = datetime(1970, 1, 1) + timedelta(seconds=timestamp)
        return dt_object.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def is_weekend(date_str):
        """
        Check if the given date is a weekend.

        Parameters
        ----------
        date_str : str
            Date string in format 'YYYY-MM-DD'

        Returns
        -------
        bool
            True if weekend, False otherwise
        """
        day = pd.to_datetime(date_str)
        return day.weekday() > 4  # 0 is Monday, 6 is Sunday

    def add_days(self, days):
        """
        Add days to the reference date.

        Parameters
        ----------
        days : int
            Number of days to add

        Returns
        -------
        str
            Resulting date string in format 'YYYY-MM-DD'
        """
        result_date = self.reference_date + timedelta(days=days)
        return result_date.strftime("%Y-%m-%d")

    def create_time_window(self, poti_type, vulnerability, start_time, end_time):
        """
        Create a single time window entry.

        Parameters
        ----------
        poti_type : str
            POI type identifier
        vulnerability : int
            Vulnerability index (0-5)
        start_time : str
            Start time in format 'YYYY-MM-DD HH:MM:SS'
        end_time : str
            End time in format 'YYYY-MM-DD HH:MM:SS'

        Returns
        -------
        dict
            Dictionary with time window data or None if error
        """
        try:
            # Validate inputs
            if not 0 <= vulnerability <= 5:
                self.log(
                    f"Invalid vulnerability value: {vulnerability}. Must be 0-5.",
                    level="error",
                )
                return None

            # Convert times to timestamps
            start_timestamp = self.to_unix_epoch(start_time)
            end_timestamp = self.to_unix_epoch(end_time)

            # Return data as dictionary
            return {
                "poti_type": poti_type,
                "vi": vulnerability,
                "ts": start_timestamp,
                "te": end_timestamp,
                "start_time": start_time,
                "end_time": end_time,
            }

        except Exception as e:
            self.log(f"Error creating time window: {str(e)}", level="error")
            return None

    def generate_from_schedule(self):
        """
        Generate time windows from predefined schedules.

        Returns
        -------
        dict of pandas.DataFrame
            Dictionary mapping POI types to their time window DataFrames
        """
        # Dictionary to store time window dataframes by POI type
        all_time_windows = {}

        # Generate time windows for each POI type
        for poti_type, schedules in self.schedules.items():
            self.log(f"Processing schedule for {poti_type}")

            time_windows_list = []

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
                    time_window = self.create_time_window(
                        poti_type=poti_type,
                        vulnerability=schedule["vulnerability"],
                        start_time=start_time,
                        end_time=end_time,
                    )

                    if time_window:
                        time_windows_list.append(time_window)

            # Convert list to DataFrame if we have any time windows
            if time_windows_list:
                all_time_windows[poti_type] = pd.DataFrame(time_windows_list)
                self.log(
                    f"Created {len(time_windows_list)} time windows for {poti_type}"
                )
            else:
                self.log(f"No time windows generated for {poti_type}", level="warning")

        self.log(
            f"Successfully generated time windows for {len(all_time_windows)} POI types"
        )
        return all_time_windows

    def get_active_time_windows(self, time_windows=None, timestamp=None):
        """
        Get all active time windows for a specific timestamp.

        Parameters
        ----------
        time_windows : dict of pandas.DataFrame, optional
            Dictionary mapping POI types to their time window DataFrames.
            If None, tries to load from default locations.
        timestamp : int, optional
            UNIX timestamp to check. Default is current time.

        Returns
        -------
        dict
            Dictionary mapping POI types to their active vulnerability indices
        """
        if timestamp is None:
            timestamp = int(datetime.now().timestamp())

        active_windows = {}

        try:
            # If no time windows provided, return empty dict
            if time_windows is None:
                self.log("No time windows provided", level="warning")
                return {}

            # Process each POI type's time windows
            for poi_type, df in time_windows.items():
                if df is not None and not df.empty:
                    # Find active windows
                    active = df[(df["ts"] <= timestamp) & (df["te"] >= timestamp)]

                    if not active.empty:
                        # Use the maximum vulnerability if multiple windows are active
                        active_windows[poi_type] = active["vi"].max()

            return active_windows

        except Exception as e:
            self.log(f"Error retrieving active time windows: {str(e)}", level="error")
            return {}

    def save(self, time_windows, path=None):
        """
        Save time windows to disk.

        Parameters
        ----------
        time_windows : dict of pandas.DataFrame
            Dictionary mapping POI types to their time window DataFrames
        path : str, optional
            Directory where time window files should be saved.
            If None, uses "./data/time_windows".

        Returns
        -------
        dict
            Dictionary with paths to saved files
        """
        if not time_windows:
            self.log("No time windows to save", level="warning")
            return {}

        # Set up output directory
        if path is None:
            time_windows_dir = os.path.abspath("./data/time_windows")
        else:
            time_windows_dir = os.path.abspath(path)

        # Create directory if it doesn't exist
        try:
            os.makedirs(time_windows_dir, exist_ok=True)
        except OSError as e:
            self.log(
                f"Failed to create directory {time_windows_dir}: {str(e)}",
                level="error",
            )
            return {}

        # Save each POI type's time windows
        saved_files = {}

        for poi_type, df in time_windows.items():
            if df is not None and not df.empty:
                try:
                    # Save to CSV
                    file_path = os.path.join(time_windows_dir, f"{poi_type}.csv")

                    # Select only the columns needed for persistence
                    save_df = df[["vi", "ts", "te"]].copy()
                    save_df.to_csv(file_path, index=False)

                    saved_files[poi_type] = file_path
                    self.log(
                        f"Saved {len(df)} time windows for {poi_type} to {file_path}"
                    )
                except Exception as e:
                    self.log(
                        f"Failed to save time windows for {poi_type}: {str(e)}",
                        level="error",
                    )

        self.log(f"Saved time windows for {len(saved_files)} POI types")
        return saved_files

    def load(self, path):
        """
        Load time windows from disk.

        Parameters
        ----------
        path : str
            Directory containing time window files

        Returns
        -------
        dict of pandas.DataFrame
            Dictionary mapping POI types to their time window DataFrames
        """
        if not os.path.exists(path) or not os.path.isdir(path):
            self.log(f"Time windows directory not found: {path}", level="error")
            return {}

        time_windows = {}

        try:
            for file_name in os.listdir(path):
                if file_name.endswith(".csv"):
                    poi_type = os.path.splitext(file_name)[0]
                    file_path = os.path.join(path, file_name)

                    # Read CSV file
                    df = pd.read_csv(file_path)

                    # Add human-readable time fields
                    if "ts" in df.columns and "te" in df.columns:
                        df["start_time"] = df["ts"].apply(self.from_unix_epoch)
                        df["end_time"] = df["te"].apply(self.from_unix_epoch)
                        df["poti_type"] = poi_type

                        time_windows[poi_type] = df
                        self.log(f"Loaded {len(df)} time windows for {poi_type}")

            self.log(f"Loaded time windows for {len(time_windows)} POI types")
            return time_windows

        except Exception as e:
            self.log(f"Error loading time windows: {str(e)}", level="error")
            return {}

    def visualize_schedule(self, time_windows=None, output_file=None):
        """
        Create an HTML visualization of the time window schedule.

        Parameters
        ----------
        time_windows : dict of pandas.DataFrame, optional
            Dictionary mapping POI types to their time window DataFrames.
            If None, uses schedules directly.
        output_file : str, optional
            Output HTML file path. If None, returns the figure object instead.

        Returns
        -------
        plotly.graph_objects.Figure or str
            Plotly figure object or path to saved HTML file
        """
        try:
            import plotly.figure_factory as ff
            import plotly.io as pio

            # Collect all time windows
            all_windows = []

            if time_windows:
                # Use provided time windows
                for poi_type, df in time_windows.items():
                    for _, row in df.iterrows():
                        # Extract day name from the start time
                        start_dt = datetime.fromtimestamp(row["ts"])
                        day_name = start_dt.strftime("%A")

                        # Format for Gantt chart
                        all_windows.append(
                            {
                                "Task": f"{poi_type} ({day_name})",
                                "Start": row["start_time"],
                                "Finish": row["end_time"],
                                "Resource": f"VI: {int(row['vi'])}",
                                "Description": f"{poi_type} - {day_name} - VI: {int(row['vi'])}",
                            }
                        )
            else:
                # Use schedules directly
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

            # Save or return figure
            if output_file:
                # Make sure directory exists
                output_dir = os.path.dirname(os.path.abspath(output_file))
                os.makedirs(output_dir, exist_ok=True)

                # Write to HTML file
                pio.write_html(fig, output_file)
                self.log(f"Schedule visualization saved to {output_file}")
                return output_file
            else:
                return fig

        except Exception as e:
            self.log(f"Error creating visualization: {str(e)}", level="error")
            return None

    def _get_vi_color(self, vulnerability):
        """
        Get color based on vulnerability index.

        Parameters
        ----------
        vulnerability : int
            Vulnerability index (0-5)

        Returns
        -------
        str
            RGB color code
        """
        colors = [
            "rgb(240,240,240)",  # 0 - Grey
            "rgb(191,255,191)",  # 1 - Light green
            "rgb(152,251,152)",  # 2 - Pale green
            "rgb(255,240,98)",  # 3 - Yellow
            "rgb(255,167,87)",  # 4 - Orange
            "rgb(255,105,97)",  # 5 - Red
        ]
        return colors[min(vulnerability, 5)]

    @classmethod
    def from_file(cls, path, verbose=True):
        """
        Create a generator and load time windows from existing files.

        Parameters
        ----------
        path : str
            Directory path containing time window files
        verbose : bool, optional
            Whether to print log messages

        Returns
        -------
        tuple
            (TimeWindowGenerator instance, dict of time window DataFrames)

        Raises
        ------
        FileNotFoundError
            If the directory doesn't exist
        ValueError
            If no valid time window files are found
        """
        if not os.path.exists(path) or not os.path.isdir(path):
            raise FileNotFoundError(f"Time windows directory not found: {path}")

        # Create a generator with default settings
        generator = cls(verbose=verbose)

        # Load the time windows
        time_windows = generator.load(path)

        if not time_windows:
            raise ValueError(f"No valid time window files found in {path}")

        return generator, time_windows
