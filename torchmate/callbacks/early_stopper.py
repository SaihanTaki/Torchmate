import numpy as np

from torchmate.callbacks import Callback


class EarlyStopper(Callback):
    """
    Callback to implement early stopping based on a monitored metric.

    This callback stops the training early if the monitored metric does not improve for a specified number
    of epochs (patience).

    Parameters:
        monitor (str): Metric to monitor for early stopping. Defaults to ``"val_loss"``.
        patience (int): Number of epochs with no improvement to wait before stopping. Defaults to ``3``.
        mode (str): One of ``"min"`` or ``"max"``. The direction to monitor the metric. For example, \
                    ``"min"`` for loss, ``"max"`` for accuracy. Defaults to ``"min"``
        min_delta (float): Minimum change in the monitored metric to qualify as an improvement. Defauls to ``0``.
        restore_best_state (bool): Whether to restore the best model state when early stopping. Defaults to ``False``.

    **Example Usage:**

    .. code-block:: python

        early_stopper = EarlyStopper(
            monitor="val_loss",
            patience=3,
            mode="min"
        )

    """

    def __init__(self, monitor="val_loss", patience=3, mode="min", min_delta=0, restore_best_state=False):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_state = restore_best_state
        self.counter = 0
        self.epoch_count = 0
        self.optimum_value = np.inf if mode == "min" else -np.inf
        self.best_model_state = None
        if self.mode not in ["min", "max"]:
            raise AttributeError("The mode parameter should be set to 'min' or 'max'")

    def on_epoch_end(self, trainer) -> None:
        """Check for early stopping conditions after each epoch and if conditions are met stops training.

        Args:
           trainer (Trainer): The trainer object.

        Returns:
           None
        """
        self.epoch_count += 1

        monitor_value = trainer.history[self.monitor][self.epoch_count - 1]
        if self.mode == "min":
            condition = (monitor_value + self.min_delta) < self.optimum_value
        elif self.mode == "max":
            condition = monitor_value > (self.optimum_value + self.min_delta)

        if condition:
            self.optimum_value = monitor_value
            self.counter = 0
            self.best_model_state = trainer.model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                trainer.early_stop = True

        if self.restore_best_state:
            trainer.model.load_state_dict(self.best_model_state)

        return None
