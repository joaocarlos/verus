import os
import signal
import time
from functools import wraps


class TimeoutException(Exception):
    """Exception raised when a function execution times out."""

    pass


def timeout_handler(signum, frame):
    """Handler for SIGALRM signal."""
    raise TimeoutException("Function execution timed out")


def with_timeout(seconds):
    """
    Decorator to apply timeout to a function.

    Args:
        seconds (int): Timeout in seconds

    Returns:
        Function wrapped with timeout functionality
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if os.name == "nt":  # Windows doesn't support SIGALRM
                # On Windows, just execute the function
                return func(*args, **kwargs)

            # On Unix systems, use SIGALRM for timeout
            original_handler = signal.getsignal(signal.SIGALRM)
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(seconds)
                result = func(*args, **kwargs)
                return result
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, original_handler)

        return wrapper

    return decorator


class Timer:
    """Utility for timing code execution."""

    def __init__(self, name=None):
        """
        Initialize the timer.

        Args:
            name (str, optional): Timer name for identification
        """
        self.name = name or "Timer"
        self.start_time = None
        self.elapsed = 0

    def __enter__(self):
        """Context manager entry point."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.stop()

    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        return self

    def stop(self):
        """Stop the timer and calculate elapsed time."""
        if self.start_time is None:
            return 0
        self.elapsed = time.time() - self.start_time
        self.start_time = None
        return self.elapsed

    def __str__(self):
        """Return a string representation of the elapsed time."""
        if self.elapsed >= 60:
            minutes = int(self.elapsed // 60)
            seconds = self.elapsed % 60
            return f"{self.name}: {minutes}m {seconds:.2f}s"
        else:
            return f"{self.name}: {self.elapsed:.2f}s"
