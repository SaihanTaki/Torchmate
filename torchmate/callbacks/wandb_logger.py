import wandb

from torchmate.callbacks import Callback


class WandbLogger(Callback):
    """Callback to log training metrics and visualizations to Weights & Biases (Wandb).

    This callback logs training metrics, visualizations, and other experiment-related data to Wandb.

    **Example Usage:**

    .. code-block:: python

        wandb_logger = WandbLogger()
        callbacks = [wandb_logger]

    """

    def __init__(self):
        super().__init__()
        self.current_epoch = 0

    def on_experiment_begin(self, trainer):
        assert wandb.run is not None, "wandb is not initialized"

    def on_epoch_end(self, trainer):
        epoch_logs = {}
        for key, value in trainer.history.items():
            epoch_logs[key] = trainer.history[key][self.current_epoch]
        wandb.log(epoch_logs)
        self.current_epoch += 1
        return None
