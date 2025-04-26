"""
    @file:              decathlon_dataset.py
    @Author:            Raphael Brodeur

    @Creation Date:     04/2025
    @Last modification: 04/2025

    @Description:       This file contains the class DecathlonDataset which is a Torch Dataset used to load Medical
                        Segmentation Decathlon data.
"""

from typing import List, Tuple

from monai.apps.datasets import DecathlonDataset as MONAIDecathlonDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    RandAffined,
    RandFlipd,
    ScaleIntensityd,
    ToTensord
)
import numpy as np
import torch
from torch.utils.data import (
    ConcatDataset,
    Dataset,
    Subset
)
from torchvision.transforms.functional import InterpolationMode, resize

from src.data.utils import DataExample


class DecathlonDataset(Dataset):
    """
    This class is a patient-level Torch Dataset for Medical Segmentation Decathlon (MSD) data used for the segmentation
    experiments in the paper.
    """

    def __init__(
            self,
            path_to_dir: str
    ):
        """
        Creates the dataset. Wraps MONAI's DecathlonDataset for simplicity.

        Parameters
        ----------
        path_to_dir : str
            The path to the directory containing the MSD tasks.
        """
        super().__init__()

        # Define transformations
        transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            ScaleIntensityd(keys=["image"]),
            ToTensord(keys=["image", "label"], track_meta=False),   # Otherwise MONAI transforms return monai.MetaTensor
        ])

        # MONAI's DecathlonDataset handles loading the data from the directory
        train_ds = MONAIDecathlonDataset(
            root_dir=path_to_dir,
            task="Task07_Pancreas",
            section="training",
            transform=transforms
        )
        val_ds = MONAIDecathlonDataset(
            root_dir=path_to_dir,
            task="Task07_Pancreas",
            section="validation",
            transform=transforms
        )

        self._dataset = ConcatDataset([train_ds, val_ds])   # Concatenate training data and validation data

    def __len__(self) -> int:
        """
        The length of the dataset.

        Returns
        -------
        length : int
            The length of the dataset.
        """
        return len(self._dataset)

    def __getitem__(self, index: int) -> DataExample:
        """
        Gets a patient from the dataset.

        Parameters
        ----------
        index : int
            The index of the patient to get.

        Returns
        -------
        item : DataExample
            A pair of the patient's image and target segmentation in dimensions (z, x, y), only slices containing
            annotations.
        """
        image = self._dataset[index]["image"]
        seg = self._dataset[index]["label"]

        # Remove tumor mask
        seg[seg == 2] = 0

        # Transpose to dims (z, x, y)
        image = image.permute(2, 1, 0)
        seg = seg.permute(2, 1, 0)

        non_zero_slices = torch.any(seg, dim=(1, 2))  # Slices where segmented organ is present

        # Only keep slices where segmented organ is present
        image = image[non_zero_slices]
        seg = seg[non_zero_slices]

        return DataExample(x=image, y=seg)


class SlicedDecathlonDataset(Dataset):
    """
    This class is a slice-level Torch Dataset for Medical Segmentation Decathlon (MSD) data used for the segmentation
    experiments in the paper.
    """

    def __init__(
            self,
            dataset: Subset,
            size: Tuple[int, int],
            apply_augmentations: bool
    ):
        """
        Creates the dataset from a Subset of DecathlonDataset.

        Parameters
        ----------
        dataset : Subset
            The Dataset containing the unsliced patients.
        size : tuple[int, int]
            The size of the slices to crop to in the format (height, width).
        apply_augmentations : bool
            Whether to apply augmentations (random horizontal flip and random affine transformation) to the data.
        """
        super().__init__()

        self._dataset = dataset
        self._size = size
        self._apply_augmentations = apply_augmentations

        self._slice_indices_mapping = self._get_slice_indices_mapping()
        self._transforms = Compose([
            RandFlipd(
                keys=["img", "seg"],
                prob=0.5,
                spatial_axis=1                  # Horizontal flip
            ),
            RandAffined(
                keys=["img", "seg"],
                prob=1.0,                       # Always sample an affine transformation
                rotate_range=5 * np.pi / 180,   # Sample rotation from U(-5, 5) degrees
                translate_range=[0.1, 0.1],     # Sample translation from U(-0.1, 0.1) in both dims (x,y)
                scale_range=[0.2, 0.2],         # Sample scaling factor from U(0.8, 1.2) in both dims
                mode="nearest",                 # Like for torchvision's RandAffine
                padding_mode="zeros"            # Like for torchvision's RandAffine
            ),
            ToTensord(keys=["img", "seg"], track_meta=False),   # Otherwise MONAI transforms return monai.MetaTensor
        ])

    def _get_slice_indices_mapping(self) -> List[Tuple[int, int]]:
        """
        Gets a mapping from slices to a patient's indices and slices.
        """
        slice_indices_mapping = []  # List containing mapping from slices to indices
        for patient_idx in range(len(self._dataset)):
            patient = self._dataset[patient_idx]

            num_slices = patient.x.shape[0]     # Number of slices

            # Add mappings to list
            slice_indices_mapping.extend(
                [(patient_idx, slice_idx) for slice_idx in range(num_slices)]
            )

        return slice_indices_mapping

    def __len__(self) -> int:
        """
        The length of the dataset (total number of slices).

        Returns
        -------
        length : int
            The length of the dataset (total number of slices).
        """
        return len(self._slice_indices_mapping)

    def __getitem__(self, index: int) -> DataExample:
        """
        Gets a slice from the dataset.

        Parameters
        ----------
        index : int
            The index of the slice to get (with respect to the dataset, not to a patient).

        Returns
        -------
        item : DataExample
            A pair of the slice's image and target segmentation.
        """
        # Get the patient and slice indices
        patient_idx, slice_idx = self._slice_indices_mapping[index]

        # Get patient tensor
        patient = self._dataset[patient_idx]

        # Get slice and add channel dimension
        image = patient.x[slice_idx].unsqueeze(0)
        seg = patient.y[slice_idx].unsqueeze(0)

        # Resize
        image = resize(img=image, size=self._size, interpolation=InterpolationMode.BILINEAR)
        seg = resize(img=seg, size=self._size, interpolation=InterpolationMode.NEAREST)

        # Augmentation (apply random horizontal flip and random affine)
        if self._apply_augmentations:
            augmented_dict = self._transforms({"img": image, "seg": seg})

            image = augmented_dict["img"]
            seg = augmented_dict["seg"]

        return DataExample(x=image, y=seg)
