"""This module provides implementations of various algorithms and modules based on research papers.

* Implementations of algorithms and models from recent research papers in various areas.
* Each function/class includes a reference to the corresponding research paper for clarity.

**Source:**

* Most of the implementations are sourced from online repositories and adapted for this module.
* Other modules are original implementations by Abdullah Saihan Taki (Author of Torchmate).

"""

from torchmate.modules.refined_self_attention_sagan import RefinedSelfAttentionSAGAN
from torchmate.modules.squeeze_and_excitation import (
    ChannelSELayer,
    ChannelSpatialSELayer,
    SELayer,
    SpatialSELayer,
)
