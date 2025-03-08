import os


class PathManager:
    """Centralized path management utility for consistent directory structures."""

    def __init__(self, output_dir=None, create_dirs=True):
        """
        Initialize the path manager.

        Args:
            output_dir (str, optional): Base output directory
            create_dirs (bool): Whether to create directories automatically
        """
        # Use provided output_dir or determine from current file location
        if output_dir:
            self.base_dir = os.path.abspath(output_dir)
        else:
            # Get the caller's file location and go up to project root
            import inspect

            caller_frame = inspect.stack()[1]
            caller_file = caller_frame.filename
            caller_dir = os.path.dirname(os.path.abspath(caller_file))

            # Navigate up to project root/data
            self.base_dir = os.path.abspath(
                os.path.join(caller_dir, "..", "..", "data")
            )

        # Define standard subdirectories
        self.subdirs = {
            "datasets": os.path.join(self.base_dir, "datasets"),
            "geojson": os.path.join(self.base_dir, "geojson"),
            "maps": os.path.join(self.base_dir, "maps"),
            "clusters": os.path.join(self.base_dir, "clusters"),
            "time_windows": os.path.join(self.base_dir, "time_windows"),
        }

        # Create directories if requested
        if create_dirs:
            self.create_all_dirs()

    def create_all_dirs(self):
        """Create all standard directories."""
        for dir_path in self.subdirs.values():
            os.makedirs(dir_path, exist_ok=True)

    def get_path(self, subdir, filename=None, place_name=None):
        """
        Get absolute path to a file in a subdirectory.

        Args:
            subdir (str): Subdirectory name (e.g., 'datasets', 'maps')
            filename (str, optional): Filename to append
            place_name (str, optional): Place name to create place-specific subdirectory

        Returns:
            str: Absolute path
        """
        if subdir not in self.subdirs:
            raise ValueError(f"Unknown subdirectory: {subdir}")

        # Get the base subdirectory path
        path = self.subdirs[subdir]

        # Add place-specific subdirectory if needed
        if place_name and subdir == "geojson":
            path = os.path.join(path, place_name)
            os.makedirs(path, exist_ok=True)

        # Add filename if provided
        if filename:
            path = os.path.join(path, filename)

        return path
