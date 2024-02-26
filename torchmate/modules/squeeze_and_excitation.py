"""A collection of squeeze and excitation (SE) layers that can be integrated into various neural network architectures.

Supports three types of SE blocks:
- Channel Squeeze and Excitation (CSE)
- Spatial Squeeze and Excitation (SSE)
- Channel and Spatial Squeeze and Excitation (CSSE)

**Credit:**
https://github.com/ai-med/squeeze_and_excitation


References:
- [Hu et al., Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
- [Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018]\
  (https://arxiv.org/abs/1803.02579)


"""

from enum import Enum

import torch


class ChannelSELayer(torch.nn.Module):
    """
    Implements the Channel Squeeze and Excitation (CSE) block as described in:

    - Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507

    Parameters:
        num_channels (int): Number of input channels.
        reduction_ratio (int, optional): Ratio to reduce the number of channels
            by in the squeeze step. Default is ``2``.

    Returns:
        torch.Tensor: Output tensor with the same dimensions as
        the input.

    Reference:
        - Paper: https://arxiv.org/abs/1709.01507
        - Implementation: https://github.com/ai-med/squeeze_and_excitation

    """

    def __init__(self, num_channels, reduction_ratio=2):
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = torch.nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = torch.nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_tensor):
        """Forward method

        Parameters:
            input_tensor: X, shape = (batch_size, num_channels, H, W)

        Returns:
            torch.Tensor: Output tensor with the same dimensions as the input.
        """

        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class SpatialSELayer(torch.nn.Module):
    """
    Implement the Spatial Squeeze and Excitation (SSE) block as described in:

    - Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks,\
    MICCAI 2018

    Parameters:
        num_channels (int): Number of input channels.

    Returns:
        torch.Tensor: Output tensor with the same dimensions as
        the input.

    Reference:
        - Paper: https://arxiv.org/abs/1803.02579
        - Implementation: https://github.com/ai-med/squeeze_and_excitation

    """

    def __init__(self, num_channels):
        super(SpatialSELayer, self).__init__()
        self.conv = torch.nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """Forward method

        Parameters:
            input_tensor: X, shape = (batch_size, num_channels, H, W)
            weights: weights for few shot learning

        Returns:
            torch.Tensor: Output tensor with the same dimensions as the input.
        """
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()

        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out = torch.nn.functional.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        # print(input_tensor.size(), squeeze_tensor.size())
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b)
        output_tensor = torch.mul(input_tensor, squeeze_tensor)
        # output_tensor = torch.mul(input_tensor, squeeze_tensor)
        return output_tensor


class ChannelSpatialSELayer(torch.nn.Module):
    """
    Implement the Concurrent Spatial and Channel Squeeze & Excitation (CSSE) block as described in:

    - Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks,\
    MICCAI 2018

    Parameters:
        num_channels (int): Number of input channels.
        reduction_ratio (int, optional): Ratio to reduce the number of channels
        by in the squeeze step. Default is ``2``.

    Returns:
        torch.Tensor: Output tensor with the same dimensions as the input.

    Reference:
        - Paper: https://arxiv.org/abs/1803.02579
        - Implementation: https://github.com/ai-med/squeeze_and_excitation


    """

    def __init__(self, num_channels, reduction_ratio=2):
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def forward(self, input_tensor):
        """Forward method

        Parameters:
            input_tensor: X, shape = (batch_size, num_channels, H, W)

        Returns:
            torch.Tensor: Output tensor with the same dimensions as the input.

        """
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor


class SELayer(Enum):
    """Squeeze and Excitation Enum Block

    Enum restricting the type of SE Blockes available. So that type checking can be adding when adding these blockes to
    a neural network.

    .. code-block:: python

        if self.se_block_type == se.SELayer.CSE.value:
            self.SELayer = se.ChannelSpatialSELayer(params['num_filters'])

        elif self.se_block_type == se.SELayer.SSE.value:
            self.SELayer = se.SpatialSELayer(params['num_filters'])

        elif self.se_block_type == se.SELayer.CSSE.value:
            self.SELayer = se.ChannelSpatialSELayer(params['num_filters'])


    Reference:
        Implementation: https://github.com/ai-med/squeeze_and_excitation
    """

    NONE = "NONE"
    CSE = "CSE"
    SSE = "SSE"
    CSSE = "CSSE"
