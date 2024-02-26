import torch


class RefinedSelfAttentionSAGAN(torch.nn.Module):
    """Implement a Refined Self-Attention module for Generative Adversarial Networks (SAGANs).

    This module has a more efficient time and space complexity of O(n) compared to the original SAGAN self-attention's
    O(n^2) complexity, making it suitable for reducing computational overhead.

    Parameters:
        in_channels (int): Number of input channels.
        kernel_size (int, optional): Size of the convolutional kernels. Defaults to ``3``.
        dilation (int, optional): Dilation factor for the convolutional layers. Defaults to ``1``.
        padding (str, optional): Padding type for the convolutional layers. Defaults to ``"same"``.
        bias (bool, optional): Whether to use bias in the convolutional layers. Defaults to ``False``.
        scale (int, optional): Scaling factor for the number of channels in the query and key projections. Defaults to ``8``.

    Reference:
        - Paper: Zheng et al., Less Memory, Faster Speed: Refining Self-Attention Module for Image Reconstruction.\
          arxiv: https://arxiv.org/abs/1905.08008
        - Implementation: This is an original implementation by the author of Torchmate.

    """

    def __init__(self, in_channels, kernel_size=3, dilation=1, padding="same", bias=False, scale=8):
        super().__init__()
        self.query = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // scale,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=bias,
        )
        self.key = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // scale,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=bias,
        )
        self.value = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=bias,
        )
        self.gamma = torch.nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def hw_flatten(x):
        return x.view(x.shape[0], x.shape[1], -1)

    def forward(self, x):
        q = self.hw_flatten(self.query(x))  # [bs, c', n]   ; n = HxW
        k = self.hw_flatten(self.key(x))  # [bs, c', n]     ; bs = batch size
        v = self.hw_flatten(self.value(x))  # [bs, c, n]
        kv = torch.bmm(k, v.permute((0, 2, 1)))  # [bs, c, c']
        norm = kv / self.hw_flatten(x).shape[2]
        attention = torch.bmm(norm.permute((0, 2, 1)), q)
        attention = attention.view_as(x)
        res = x + attention
        return res
