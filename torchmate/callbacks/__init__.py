"""
**Callbacks: Hooks into the Neural Network Training Process**

In the realm of neural network training, callbacks play a crucial role by enabling you to intercept events and execute
custom actions at various stages. This mechanism empowers you to tailor the training process to your specific needs,
enhance learning performance, and gain valuable insights from the training data.

**Why Use Callbacks?**

Callbacks offer a plethora of benefits, including:

- Monitoring Progress: Track metrics like loss and accuracy, visualize them dynamically, and even halt \
  training early if certain criteria are met.
- Saving Checkpoints: Regularly store snapshots of the model's state at different points in time, allowing \
  you to resume training from a saved point or experiment with different hyperparameters.
- Early Stopping: Prevent overfitting by automatically stopping training when validation performance starts to decline.
- Logging: Record training details, metrics, and other information for analysis and evaluation.
- Custom Actions: Implement specialized techniques or integrate with external services during training.


"""

from torchmate.callbacks.callback import Callback
from torchmate.callbacks.csv_logger import CSVLogger
from torchmate.callbacks.early_stopper import EarlyStopper
from torchmate.callbacks.gradient_accumulator import GradientAccumulator
from torchmate.callbacks.gradient_clipper import GradientClipper
from torchmate.callbacks.model_checkpoint import ModelCheckpoint
from torchmate.callbacks.wandb_logger import WandbLogger
from torchmate.callbacks.wandb_model_checkpoint import WandbModelCheckpoint

# __all__ = ["Callback",
#            "GradientAccumulator",
#            "GradientClipper",
#            "ModelCheckpoint",
#            "CSVLogger",
#            "EarlyStopper",
#            "WandbLogger",
#            "WandbModelCheckpoint"
#            ]
