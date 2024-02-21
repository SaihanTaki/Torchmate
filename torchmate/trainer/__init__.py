"""
The Trainer class is a customizable tool designed to streamline the training, evaluation,\
and prediction processes for PyTorch models.

With a comprehensive set of features and flexible functionalities, it empowers users to efficiently train and assess their
machine learning models with ease. At its core, the Trainer class encapsulates essential components necessary for model training,
including the model architecture, data loaders for training, validation, and optionally testing datasets, loss function, optimizer,
and scheduling mechanisms. It also provides hooks for incorporating custom metrics and callbacks, offering extensibility for
tailored evaluation and logging during training.

**Key Features:**

- `Mixed Precision Training:` Enables faster training and potentially better generalization through automatic mixed precision (AMP).
- `Gradient Penalty:` Helps stabilize training in generative models like GANs.
- `Callback System:` Supports custom actions at different training stages (e.g., logging, early stopping, model saving).
- `Customizable Training and Validation:` Allows for flexible configuration of loss functions, metrics, and datasets.
- `Experiment Tracking:` For better experiment tracking, ditch spreadsheets and use dedicated tools like Weights & Biases
   or TensorBoard.These platforms track training, log metrics, visualize results, and organize experiments, saving you time
   and aiding collaboration. Torchmate's Trainer integrates them seamlessly via callbacks.
"""

from torchmate.trainer.trainer import Trainer

# __all__ = ["Trainer"]
