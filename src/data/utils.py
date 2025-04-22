"""
    @file:              utils.py
    @Author:            Raphael Brodeur

    @Creation Date:     04/2025
    @Last modification: 04/2025

    @Description:       This file contains utility functions for data processing.
"""

from typing import NamedTuple

from torch import Tensor


class DataExample(NamedTuple):
    """
    Stores an example's image and segmentation.

    Elements
    --------
    x : Tensor
        The example's image.
    y : Tensor
        The example's target segmentation.
    """
    x: Tensor
    y: Tensor
