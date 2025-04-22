"""
    @file:              decathlon_dataset.py
    @Author:            Raphael Brodeur

    @Creation Date:     04/2025
    @Last modification: 04/2025

    @Description:       This file contains the class DecathlonDataset which is a Torch Dataset used to load Medical
                        Decathlon Segmentation data.
"""

from torch.utils.data import Dataset

from src.data.utils import DataExample


class DecathlonDataset(Dataset):
    """
    This class is a Torch Dataset for Medical Decathlon Segmentation data used for the segmentation experiments in the
    paper.
    """

    def __init__(self):
        """
        Creates the dataset.

        Parameters
        ----------
        ...
        """
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
