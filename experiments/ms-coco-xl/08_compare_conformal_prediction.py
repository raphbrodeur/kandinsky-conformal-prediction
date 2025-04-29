"""
    @file:              07_imagewise_conformal_prediction.py
    @Author:            Raphael Brodeur

    @Creation Date:     04/2025
    @Last modification: 04/2025

    @Description:       This file contains the script to perform image-wise conformal prediction for the MS-COCO-XL
                        experiment from the paper.
"""

import matplotlib.pyplot as plt
from monai.utils import set_determinism
import torch
from scipy.cluster.vq import kmeans
from torch.utils.data import DataLoader, random_split

from src.data import COCODataset
from src.models import UNetPaper

if __name__ == "__main__":
    # Set random seed
    set_determinism(seed=1010710)

    # Hyperparameters
    num_training_samples = 678
    num_calibration_samples = 20000
    num_testing_samples = 2869

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_workers = 0

    # Dataset
    ds = COCODataset(
        path_to_dir="C:/Users/Labo/Desktop/datasets/coco_2017_seg/learning",
        size=[320, 240],
        apply_augmentations=False
    )

    # Split dataset into training and calibration sets
    train_ds, calib_ds, left_over_data = random_split(
        dataset=ds,
        lengths=[
            num_training_samples,                                       # 678
            num_calibration_samples,                                    # 20000
            len(ds) - num_training_samples - num_calibration_samples    # 43437
        ]
    )

    # Take 2869 test samples from left_over_data
    test_ds, _ = random_split(
        dataset=left_over_data,
        lengths=[
            num_testing_samples,
            len(left_over_data) - num_testing_samples,
        ]
    )

    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Model
    net = UNetPaper(
        in_channels=3,
        out_channels=1,
        channels=128
    ).to(device)

    # Load model weights
    net.load_state_dict(torch.load("./saved_params/model_params.pt", map_location=device))

    # Load non-conformity curves
    pixelwise_nc_curves = torch.load("./saved_non_conformity_curves/pixelwise_non_conformity_curves.pt", map_location=device)
    imagewise_nc_curves = torch.load("./saved_non_conformity_curves/imagewise_non_conformity_curves.pt", map_location=device)
    kmeans_nc_curves = torch.load("./saved_non_conformity_curves/kmeans_non_conformity_curves.pt", map_location=device)
    genann_nc_curves = torch.load("./saved_non_conformity_curves/gen_ann_non_conformity_curves.pt", map_location=device)
    fcc

