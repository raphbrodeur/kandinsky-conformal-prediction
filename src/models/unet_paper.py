"""
    @file:              unet_paper.py
    @Author:            Raphael Brodeur

    @Creation Date:     04/2025
    @Last modification: 04/2025

    @Description:       This file contains the basic 2D UNet model used for experiments in the paper "Kandinsky
                        Conformal Prediction: Efficient Calibration of Image Segmentation Algorithms" by Joren
                        Brunekreef.
"""

from torch import cat, Tensor
from torch.nn import (
    Conv2d,
    InstanceNorm2d,
    Module,
    ModuleList,
    ReLU,
    Sequential
)
from torch.nn.functional import interpolate, max_pool2d


class DoubleConvBlock(Sequential):
    """
    This class is the double convolution block used by the paper for the down sampling and up sampling of the UNet
    used in the paper. First convolution affects the number of channels.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ):
        """
        Initializes the module.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        """
        super().__init__()

        # First conv + norm + act
        conv1 = Sequential()

        conv1.add_module(
            name="Conv",
            module=Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )
        conv1.add_module(name="Norm", module=InstanceNorm2d(out_channels))
        conv1.add_module(name="Act", module=ReLU())

        self.add_module(name="Conv1", module=conv1)

        # Second conv + norm + act
        conv2 = Sequential()

        conv2.add_module(
            name="Conv",
            module=Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )
        conv2.add_module(name="Norm", module=InstanceNorm2d(out_channels))
        conv2.add_module(name="Act", module=ReLU())

        self.add_module(name="Conv2", module=conv2)


class UNetPaper(Module):
    """
    This class is a 2D UNet.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            channels: int
    ):
        """
        Initializes the model.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        channels : int
            The number of channels output by the first down sampling layer. The number of channels is then doubled by
            each of the 3 other down sampling layers.
        """
        super().__init__()

        # Down-sampling path
        self.down_sample_path = ModuleList(
            [DoubleConvBlock(in_channels=in_channels, out_channels=channels)]
        )
        for _ in range(3):
            self.down_sample_path.append(
                DoubleConvBlock(
                    in_channels=channels,
                    out_channels=2 * channels
                )
            )
            channels = 2 * channels

        # UNet bottleneck
        self.bottleneck = DoubleConvBlock(in_channels=channels, out_channels=channels)

        # Up-sampling path
        self.up_sample_path = ModuleList()
        for _ in range(3):
            self.up_sample_path.append(
                DoubleConvBlock(
                    in_channels=2 * channels,
                    out_channels=channels // 2
                )
            )
            channels = channels // 2

        self.up_sample_path.append(
            DoubleConvBlock(
                in_channels=2 * channels,
                out_channels=channels
            )
        )

        # Final down-channeling
        self.final_conv = Sequential(
            Conv2d(
                in_channels=channels,
                out_channels=channels // 2,
                kernel_size=1
            ),
            Conv2d(
                in_channels=channels // 2,
                out_channels=out_channels,
                kernel_size=1
            ),
            Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        y : Tensor
            The output tensor.
        """
        stack = []  # List of outputs for skip connections

        # Down-sampling
        for module in self.down_sample_path:
            x = module(x)                                   # Forward pass DoubleConv + norm + act
            stack.append(x)                                 # Store in list for skip connections
            x = max_pool2d(input=x, kernel_size=2)          # Max pooling for down-sizing

        # Bottleneck of UNet
        x = self.bottleneck(x)

        # Up-sampling
        for module in self.up_sample_path:
            x = interpolate(input=x, scale_factor=2, mode="bilinear", align_corners=False)
            x = module(
                cat([x, stack.pop()], dim=1)
            )

        # Final down-channeling
        x = self.final_conv(x)

        return x
