"""
    @file:              coco_dataset.py
    @Author:            Raphael Brodeur

    @Creation Date:     04/2025
    @Last modification: 04/2025

    @Description:       This file contains the class COCODataset which is the Dataset from the paper used to load data
                        from MS-COCO.
"""

from pathlib import Path
from typing import Tuple

from monai.transforms import (
    Compose,
    RandAffined,
    RandFlipd,
    ToTensord
)
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import (
    InterpolationMode,
    pad,
    resize
)

from src.data.utils import DataExample


class COCODataset(Dataset):
    """
    This class is a Torch Dataset for MS-COCO data used for the segmentation experiments in the paper.
    """

    def __init__(
            self,
            path_to_dir: str,
            size: Tuple[int, int],
            apply_augmentations: bool
    ):
        """
        Creates the dataset.

        Parameters
        ----------
        path_to_dir : str
            The path to the directory containing the data and labels.
        size : tuple[int, int]
            The size of the images to crop to in the format (height, width).
        apply_augmentations : bool
            Whether to apply augmentations (random horizontal flip and random affine transformation) to the data.
        """
        super().__init__()

        self._path_to_dir = Path(path_to_dir)
        self._size = size
        self._apply_augmentations = apply_augmentations

        self._coco_labels = COCO(self._path_to_dir / "labels.json")
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
            ToTensord(keys=["img", "seg"], track_meta=False),
        ])

    def __len__(self) -> int:
        """
        The length of the dataset.

        Returns
        -------
        length : int
            The length of the dataset.
        """
        return len(self._coco_labels.imgs)

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
        # Get image
        image_pil = Image.open(
            self._path_to_dir / "data" / self._coco_labels.loadImgs(index)[0]["file_name"]
        ).convert("RGB")

        # Get segmentation
        annotations_id = self._coco_labels.getAnnIds(imgIds=index, catIds=[50], iscrowd=None)   # 50 => "persons"
        annotations = self._coco_labels.loadAnns(annotations_id)

        w, h = image_pil.size           # Image's size
        seg_mask = np.zeros((h, w, 1))  # Segmentation mask with same size as image
        for ann in annotations:
            binary_mask = np.asarray(self._coco_labels.annToMask(ann))

            seg_mask[:, :, 0] = np.logical_or(seg_mask[:, :, 0], binary_mask)

        # Convert to tensor
        image = torch.tensor(np.asarray(image_pil).transpose(2, 0, 1), dtype=torch.float32)    # Image tensor (3, H, W)
        seg = torch.tensor(seg_mask.transpose(2, 0, 1), dtype=torch.float32)    # Segmentation tensor (1, H, W)

        # Adjust resize dims (some images are vertical, others are horizontal)
        aspect_ratio = w / h
        if w > h:
            w = self._size[0]
            h = int(w / aspect_ratio)
        else:
            h = self._size[1]
            w = int(h * aspect_ratio)

        # Resize
        image = resize(img=image, size=(h, w), interpolation=InterpolationMode.BILINEAR)
        seg = resize(img=seg, size=(h, w), interpolation=InterpolationMode.NEAREST)

        # Pad
        pad_w = self._size[0] - w
        pad_h = self._size[1] - h

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        image = pad(img=image, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=0)
        seg = pad(img=seg, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=0)

        # Augmentation (apply random horizontal flip and random affine)
        if self._apply_augmentations:
            augmented_dict = self._transforms({"img": image, "seg": seg})

            image = augmented_dict["img"]
            seg = augmented_dict["seg"]

        return DataExample(x=image, y=seg)
