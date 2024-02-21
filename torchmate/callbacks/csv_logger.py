import csv
import os

from torchmate.callbacks import Callback


class CSVLogger(Callback):
    """Logs training history to a CSV file.

    Args:
       filename (str): The path to the CSV file to log to.
       separator (str, optional): The field separator to use in the CSV file. Defaults to ``","``.
       append (bool, optional): Whether to append to an existing file or create a new one. Defaults to ``False``.

    **Example Usage:**

    .. code-block:: python

        csv_logger = CSVLogger(filename="logs/log.csv")
        callbacks = [csv_logger]


    """

    def __init__(self, filename: str, separator: str = ",", append: bool = False):
        super().__init__()
        self.filename = filename
        self.separator = separator
        self.append = append
        self.current_epoch = 0

    def on_experiment_begin(self, trainer) -> None:
        """Initialize the logging process at the beginning of the experiment.

        Args:
           trainer (Trainer): The trainer object.

        Returns:
           None
        """

        file_dir = os.path.dirname(self.filename)
        if not os.path.exists(file_dir) and file_dir != "":
            os.makedirs(file_dir)

        open_flag = "a" if self.append else "w"
        fieldnames = ["Epoch"] + list(trainer.history.keys())
        self.csvfile = open(self.filename, open_flag, newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.csvfile, fieldnames=fieldnames, delimiter=self.separator)

        if not self.append:
            self.writer.writeheader()
            self.csvfile.flush()

        return None

    def on_epoch_end(self, trainer) -> None:
        """Log training metrics after each epoch.

        Args:
           trainer (Trainer): The trainer object.

        Returns:
           None
        """

        self.current_epoch += 1

        epoch_logs = {"Epoch": self.current_epoch}
        for key, value in trainer.history.items():
            epoch_logs[key] = trainer.history[key][self.current_epoch - 1]

        self.writer.writerow(epoch_logs)
        self.csvfile.flush()
        return None

    def on_experiment_end(self, trainer) -> None:
        """Close the CSV file at the end of the experiment.

        Args:
           trainer (Trainer): The trainer object.

        Returns:
           None
        """

        self.csvfile.close()

        return None
