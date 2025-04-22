"""
    @file:              coco_dataset.py
    @Author:            Raphael Brodeur

    @Creation Date:     04/2025
    @Last modification: 04/2025

    @Description:       This file contains the class COCODataset which is the Dataset from the paper used to load data
                        from MS-COCO.
"""

from pathlib import Path
from typing import NamedTuple, Tuple

from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from torch import Tensor, tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode, pad, resize


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
            size: Tuple[int, int]
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

        self._path_to_dir = Path(path_to_dir)
        self._size = size

        self._coco_labels = COCO(self._path_to_dir / "labels.json")

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
        image = tensor(np.asarray(image_pil).transpose(2, 0, 1))    # Create image tensor (3, H, W)
        seg = tensor(seg_mask.transpose(2, 0, 1))                   # Create segmentation tensor (1, H, W)

        # Resize and pad
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

        # Transforms
        ...

        return DataExample(x=image, y=seg)
