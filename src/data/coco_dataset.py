"""
    @file:              coco_dataset.py
    @Author:            Raphael Brodeur

    @Creation Date:     04/2025
    @Last modification: 04/2025

    @Description:       This file contains the class COCODataset which is the Dataset from the paper used to load data
                        from MS-COCO.
"""

from typing import NamedTuple

from torch import Tensor
from torch.utils.data import Dataset


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


class COCODataset(Dataset):
    """
    This class is a Torch Dataset for MS-COCO data used for the segmentation experiments in the paper.
    """

    def __init__(
            self,
            path_to_dir: str,
            size: tuple[int, int]
    ):
        """
        Creates the dataset.

        Parameters
        ----------
        path_to_dir : str
            The path to the directory containing the data and labels.
        size : tuple[int, int]
            The size of the images to crop to in the format (height, width).
        """
        super().__init__()

        pass

    def __len__(self) -> int:
        """
        The length of the dataset.

        Returns
        -------
        length : int
            The length of the dataset.
        """
        pass

    def __getitem__(self, index: int) -> DataExample:
        """
        Gets an item from the dataset.

        Parameters
        ----------
        index : int
            The index of the item to get.

        Returns
        -------
        item : DataExample
            A pair of the item's image and target segmentation.
        """
        pass
