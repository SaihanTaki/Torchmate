import time


class ProgressBar:
    """A progress bar for tracking the progress of iterative tasks.

    Displays a progress bar with percentage completion, elapsed time, estimated time remaining,
    and optional verbose text.

    **Example usage:**

    .. code-block:: python

        no_iter = 100
        pb = ProgressBar(total=no_iter, prefix="Processing", bar_length=50)
        for i in range(no_iter):
            pb.update(i + 1, message=f"Processing item {i + 1}")

    """

    def __init__(self, total: int, bar_length: int = 30, fill: str = "=", prefix: str = "") -> None:
        """Initialize the instance of ProgressBar class.

        Args:
            total (int): Total number of iterations. Defaults to None.
            bar_length (int, optional): Length of the progress bar. Defaults to ``30``.
            fill (str, optional): Character used to fill the progress bar. Defaults to ``'='``.
            prefix (str, optional): A text to display before the progress bar. Defaults to empty string ``""``.
        """

        self.total = total
        self.length = bar_length
        self.fill = fill
        self.prefix = prefix + ": " if prefix else ""
        self.start_time = time.time()
        self._prev_len = 0

    def update(self, iteration: int, message: str = "") -> None:
        """
        Update the progress bar with current iteration and optional message text.

        Args:
            iteration (int): Current iteration number.
            message (str, optional): Optional text to display. Defaults to empty string.

        """

        # Calculate progress and format bar
        percent = ("{0:.1f}").format(100 * (iteration / float(self.total)))
        filled_length = int(self.length * iteration // self.total)
        bar = self.fill * filled_length + "-" * (self.length - filled_length)

        # Calculate elapsed and estimated time
        elapsed_time = time.time() - self.start_time
        ET = self.format_time(elapsed_time)
        estimated_time_remaining = (elapsed_time / iteration) * (self.total - iteration)
        ETR = self.format_time(estimated_time_remaining)

        # Print progress bar with appropriate text
        if iteration == self.total:
            step_time = self.format_time(elapsed_time / self.total)
            info = f"\r{self.prefix}{iteration}/{self.total} [{bar}] {percent}% | ET: {ET} | {step_time}\\step | "
            text = info + message
            end = "\n"
        else:
            info = f"\r{self.prefix}{iteration}/{self.total} [{bar}] {percent}% | ET: {ET}| ETR : {ETR} | "
            text = info + message
            end = "\r"
        print(text, end=end, flush=True)

    @staticmethod
    def format_time(duration_seconds: float) -> str:
        """
        Format a duration in seconds into a human-readable string.

        Args:
            duration_seconds (float): Duration in seconds.

        Returns:
            str: Formatted time string (e.g., "1m 23s", "45.67s", "2h 35m").

        **Example usage:**

        .. code-block:: python

            formatted_time = ProgressBar.format_time(65.5)  # Returns "1:05"

        """

        if duration_seconds < 1:
            formated_time = f"{duration_seconds * 1e3:.0f}ms"
        elif duration_seconds < 60:
            formated_time = f"{duration_seconds:.2f}s"
        elif duration_seconds < 3600:
            minutes = int(duration_seconds // 60)
            seconds = int(duration_seconds % 60)
            formated_time = f"{minutes:02d}:{seconds:02d}min"
        else:
            hours = int(duration_seconds // 3600)
            minutes = int((duration_seconds % 3600) // 60)
            seconds = int(duration_seconds % 60)
            formated_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return formated_time
