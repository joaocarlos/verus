class TimeoutException(Exception):
    """Exception raised when a function execution times out."""

    pass


def timeout_handler(signum, frame):
    """Handler for SIGALRM signal."""
    raise TimeoutException("Function execution timed out")


def with_timeout(seconds=10, error_message="Function call timed out"):
    """
    Decorator to apply a timeout to a function.

    Args:
        seconds (int): Timeout in seconds
        error_message (str): Message to display when timeout occurs

    Returns:
        decorated function

    Raises:
        TimeoutException: If function execution exceeds the timeout

    Example:
        @with_timeout(5)
        def slow_function():
            time.sleep(10)  # This will cause a timeout
    """
    # Rest of the implementation...


class Timer:
    """
    Simple context manager for timing code execution.

    Example:
        with Timer("My operation"):
            time.sleep(1)
            # Will print "My operation took 1.00 seconds"
    """

    def __init__(self, name=None):
        """
        Initialize the timer.

        Args:
            name (str, optional): Name of the operation being timed. Defaults to None.
        """
        self.name = name
        self.start_time = None

    def __enter__(self):
        """Start timing when entering the context."""
        import time

        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Print elapsed time when exiting the context."""
        import time

        elapsed = time.time() - self.start_time

        if self.name:
            print(f"{self.name} took {elapsed:.2f} seconds")
        else:
            print(f"Operation took {elapsed:.2f} seconds")
