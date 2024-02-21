import os

import numpy as np
import torch
import wandb

from torchmate.callbacks import Callback


class ModelCheckpoint(Callback):
    """
    Callback to save model checkpoints.

    This callback saves model checkpoints at specified intervals and under certain conditions such as best
    validation loss or metrics.

    Parameters:
        checkpoint_dir (str): Directory to save the checkpoints.
        monitor (str, optional): Metric to monitor for saving the best model. Defaults to ``"val_loss"``.
        mode (str,  optional): One of ``"min"`` or ``"max"``. The direction to monitor the metric.\
                                For example, ``"min"`` for loss, ``"max"`` for accuracy. Defaults to ``"min"``.
        min_delta (float, optional): Minimum change in the monitored metric to qualify as an improvement. Defaults to ``0.0``.
        save_frequency (int, optional): Frequency of saving checkpoints (epochs). Defaults to ``1``.
        save_best_only (bool, optional): Whether to save only the best model based on the monitored metric. Defaults to ``True``.
        save_last (bool, optional): Whether to save the model checkpoint for the last epoch. Defaults to ``True``.
        save_state_dict_only (bool, optional): Whether to save only the model's state dictionary instead of the whole model.\
        Defaults to ``True``.

            - If ``True``:
                * Saves a dictionary containing `epoch`, `model_state_dict`, and `optimizer_state_dict`.
                * Requires manually creating model and optimizer instances before loading the dictionary.
                * Only the model's weights and biases are loaded from `model_state_dict`.
            - If ``False``:
                * Saves the entire model object.

    **Example Usage:**

    .. code-block:: python

        checkpoint_callback = ModelCheckpoint(
            checkpoint_dir="checkpoints",
            monitor="val_loss",
            mode="min",
            save_best_only = False,
            save_frequency=2,
        )
        callbacks = [checkpoint_callback]

    """

    def __init__(
        self,
        checkpoint_dir,
        monitor="val_loss",
        mode="min",
        min_delta=0.0,
        save_frequency=1,
        save_best_only=True,
        save_state_dict_only=True,
        save_last=True,
    ):

        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.save_frequency = save_frequency
        self.save_best_only = save_best_only
        self.save_state_dict_only = save_state_dict_only
        self.save_last = save_last
        self.epoch_count = 0
        self.optimum_value = np.inf if mode == "min" else -np.inf
        self.best_checkpoint_path = None

    def on_epoch_end(self, trainer):
        self.epoch_count += 1

        checkpoint_dir = os.path.join(self.checkpoint_dir, "ckpts")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        if self.save_state_dict_only:
            model = {
                "epoch": self.epoch_count,
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
            }
        else:
            model = trainer.model

        monitor_value = trainer.history[self.monitor][self.epoch_count - 1]

        if self.mode == "min":
            condition = (monitor_value + self.min_delta) < self.optimum_value
        elif self.mode == "max":
            condition = monitor_value > (self.optimum_value + self.min_delta)

        if condition:
            self.optimum_value = monitor_value

        if self.save_best_only and condition:
            # Delete the previous best checkpoint if it exists
            if self.best_checkpoint_path:
                os.remove(self.best_checkpoint_path)
            checkpoint_name = "model_best.pt"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            torch.save(model, checkpoint_path)
            self.best_checkpoint_path = checkpoint_path
        elif not self.save_best_only:
            if self.epoch_count % self.save_frequency == 0:
                checkpoint_name = f"model_epoch_{self.epoch_count}_{self.monitor}_{monitor_value:.5f}.pt"
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
                torch.save(model, checkpoint_path)

        if self.save_last and self.epoch_count == trainer.num_epochs:
            filepath = os.path.join(checkpoint_dir, "model_last.pt")
            torch.save(model, filepath)
