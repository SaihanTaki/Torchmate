import math
import re
from typing import Dict, Optional, Type

import matplotlib.pyplot as plt


class HistoryPlotter:
    """
    A class for plotting training and validation metrics, including loss, learning rate, and any additional \
    metrics provided by the training history.

    Args:
        trainer (Trainer, optional): An instance of ``torchmate.trainer.Trainer`` containing the training history to be plotted.
        history (Dict, optional): A dictionary containing the training history data, with keys such as \
        ``loss``, ``lr``, and other metric names.

    Raises:
        ValueError: If both ``trainer`` and ``history`` are None.

    Notes:
        If both ``trainer`` and ``history`` are provided, the training history from ``trainer`` will take precedence.


    **Example usage:**

    .. code-block:: python

        # Using a Trainer object:
        trainer = Trainer(...)
        history = trainer.fit()
        plotter = Plotter(trainer=trainer)
        plotter.plot_all()


        # Using a history dictionary returned from trainer.fit()
        trainer = Trainer(...)
        history = trainer.fit()
        plotter = Plotter(history=history)
        plotter.plot_all()

        # Using a history dictionary:
        history = {"loss": [...], "lr": [...], "some_metric": [...]}
        plotter = Plotter(history=history)
        plotter.plot_all()

    """

    def __init__(self, trainer=None, history=None):

        if trainer is None and history is None:
            raise ValueError("Either 'trainer' or 'history' attribute must be provided.")

        if trainer:
            self.history = trainer.history
            self.metrics = [metric.__name__ for metric in trainer.metrics]
        else:
            metrics = []
            for k, v in history.items():
                if k not in ["Epoch", "loss", "lr"] and not k.startswith("val_"):
                    metrics.append(k)
            self.metrics = metrics if len(metrics) > 0 else None
            self.history = history

    @staticmethod
    def format_to_space_capitalized(text):
        """
        Convert a string to space-separated capitalized words, handling various formats.

        Args:
            text: A string in any format (snake_case, camelCase, space-separated, etc.).

        Returns:
            A string with words separated by spaces and capitalized.
        """
        text = re.sub("_", " ", text)
        text = re.sub(r"(?<!^)([A-Z][a-z]*)", r" \1", text)
        text = " ".join(text.split())
        text = text.title()
        return text

    def plot_all(self):
        """
        Plot loss, learning rate and all available metrics.
        """
        if self.metrics is not None:
            num_col = 2
            num_row = math.ceil((len(self.metrics) + 2) / 2)
            fig, ax = plt.subplots(num_row, num_col, figsize=(16, num_row * 5))
            ax = ax.ravel()
        else:
            num_row = 1
            num_col = 2
            fig, ax = plt.subplots(num_row, num_col, figsize=(16, num_row * 5))
            ax = ax.ravel()

        ax[0].plot([None] + self.history["loss"], "o-")
        ax[0].plot([None] + self.history["val_loss"], "o-")
        ax[0].legend(["Training", "Validation"], loc=0)
        ax[0].set_title("Training & Validation Loss", fontsize=20)
        ax[0].set_xlabel("Epoch", fontsize=16)
        ax[0].set_ylabel("Loss", fontsize=16)
        x_ticks = list(range(1, len(self.history["loss"]) + 1))
        ax[0].set_xticks(x_ticks)
        ax[0].grid(True)

        ax[1].plot([None] + self.history["lr"], "o-")
        ax[1].set_title("Learning Rate", fontsize=20)
        ax[1].set_xlabel("Epoch", fontsize=16)
        ax[1].set_ylabel("Learning Rate", fontsize=16)
        x_ticks = list(range(1, len(self.history["lr"]) + 1))
        ax[1].set_xticks(x_ticks)
        ax[1].grid(True)

        if self.metrics is not None:
            for ix, metric in enumerate(self.metrics):
                ax[ix + 2].plot([None] + self.history[metric], "o-")
                ax[ix + 2].plot([None] + self.history[f"val_{metric}"], "o-")
                ax[ix + 2].legend(["Training", "Validation"], loc=0)
                metric_name = self.format_to_space_capitalized(metric)
                ax[ix + 2].set_title(f"Training & Validation {metric_name}", fontsize=20)
                ax[ix + 2].set_xlabel("Epoch", fontsize=16)
                ax[ix + 2].set_ylabel(f"{metric_name}", fontsize=16)
                x_ticks = list(range(1, len(self.history[metric]) + 1))
                ax[ix + 2].set_xticks(x_ticks)
                ax[ix + 2].grid(True)

        fig.tight_layout()
        plt.show()
        return None

    def plot_metrics(self):
        """
        Plot the training and validation metrics.
        """
        if self.metrics is None:
            print("There are no metrics to plot!")
            return None

        num_col = 2
        num_row = math.ceil((len(self.metrics)) / 2)
        fig, ax = plt.subplots(num_row, num_col, figsize=(16, num_row * 5))
        ax = ax.ravel()

        for ix, metric in enumerate(self.metrics):
            ax[ix].plot([None] + self.history[metric], "o-")
            ax[ix].plot([None] + self.history[f"val_{metric}"], "o-")
            ax[ix].legend(["Training", "Validation"], loc=0)
            metric_name = self.format_to_space_capitalized(metric)
            ax[ix].set_title(f"Training & Validation {metric_name}", fontsize=20)
            ax[ix].set_xlabel("Epoch", fontsize=16)
            ax[ix].set_ylabel(f"{metric_name}", fontsize=16)
            x_ticks = list(range(1, len(self.history[metric]) + 1))
            ax[ix].set_xticks(x_ticks)
            ax[ix].grid(True)
        fig.tight_layout()
        plt.show()
        return None

    def plot_loss(self):
        """
        Plot the training and validation loss.
        """
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot([None] + self.history["loss"], "o-")
        ax.plot([None] + self.history["val_loss"], "o-")
        ax.legend(["Training", "Validation"], loc=0)
        ax.set_title("Training & Validation Loss", fontsize=16)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        x_ticks = list(range(1, len(self.history["loss"]) + 1))
        ax.set_xticks(x_ticks)
        ax.grid(True)
        fig.tight_layout()
        plt.show()
        return None

    def plot_lr(self):
        """
        Plot the learning rate over epochs.
        """
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot([None] + self.history["lr"], "o-")
        ax.set_title("Learning Rate", fontsize=16)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Learning Rate", fontsize=12)
        x_ticks = list(range(1, len(self.history["lr"]) + 1))
        ax.set_xticks(x_ticks)
        ax.grid(True)
        fig.tight_layout()
        plt.show()
        return None
