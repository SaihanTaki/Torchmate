from typing import Optional, Union

import torch

from torchmate.callbacks import Callback


class GradientClipper(Callback):
    """
    GradientClipper class for clipping gradients during the training process.

    Args:
        method (str): The gradient clipping method to use.
            Supported methods are "clip_by_value" and "clip_by_norm".
        clip_value (Optional[float]): Maximum allowed value of gradients for
            clip_by_value method. Required if method is "clip_by_value".
        max_norm (Optional[float]): Maximum allowed norm of gradients for
            clip_by_norm method. Required if method is "clip_by_norm".

    Raises:
        ValueError: If the provided method is not supported.
        ValueError: If clip_value is not provided when method is "clip_by_value".
        ValueError: If max_norm is not provided when method is "clip_by_norm".

    **Example Usage:**

    .. code-block:: python

        gradient_clipper = GradientClipper(method="clip_by_value", clip_value=1.0)
        callbacks = [gradient_clipper]

    """

    def __init__(
        self,
        method: str,
        clip_value: Optional[float] = None,
        max_norm: Optional[float] = None,
    ) -> None:

        supported_methods = ["clip_by_value", "clip_by_norm"]
        if method not in supported_methods:
            raise ValueError(
                f"Unsupported gradient clipping method '{method}'. Supported methods are: {', '.join(supported_methods)}"
            )

        self.method = method
        self.clip_value = clip_value
        self.max_norm = max_norm

        if self.method == "clip_by_value" and self.clip_value is None:
            raise ValueError(f"clip_value must be provided when method is '{method}'")

        if self.method == "clip_by_norm" and self.max_norm is None:
            raise ValueError(f"max_norm must be provided when method is '{method}'")

    def on_backward_end(self, trainer) -> None:
        """Clips gradients at the end of the backward pass."""
        trainer.scaler.unscale_(trainer.optimizer)

        if self.method == "clip_by_norm":
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=self.max_norm)

        elif self.method == "clip_by_value":
            torch.nn.utils.clip_grad_value_(trainer.model.parameters(), clip_value=self.clip_value)
