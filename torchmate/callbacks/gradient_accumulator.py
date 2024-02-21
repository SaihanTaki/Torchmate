from typing import Optional

from torchmate.callbacks import Callback


class GradientAccumulator(Callback):
    """
    Callback to accumulate gradients for gradient accumulation.

    This callback accumulates gradients over a specified number of batches and performs an update
    when the specified number of batches is reached.

    Parameters:
        num_accum_steps (int): Number of accumulation steps before performing a parameter update.

    **Example Usage:**

    .. code-block:: python

        gradient_accumulator = GradientAccumulator(num_accum_steps=4)
        callbacs = [gradient_accumulator]

    """

    def __init__(self, num_accum_steps: Optional[int] = None):
        super().__init__()
        self.batch_count = 0
        self.num_accum_steps = num_accum_steps

    def on_experiment_begin(self, trainer):
        trainer.update_params = False
        if self.num_accum_steps is not None:
            trainer.accumulation_steps = self.num_accum_steps

    def on_train_batch_begin(self, trainer):
        self.batch_count += 1
        if (self.batch_count % trainer.accumulation_steps == 0) or (self.batch_count == len(trainer.train_dataloader)):
            trainer.update_params = True
        else:
            trainer.update_params = False

    # def on_train_batch_end(self,trainer):
    #     self.batch_count += 1
