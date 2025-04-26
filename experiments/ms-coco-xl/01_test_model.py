"""
    @file:              01_test_model.py
    @Author:            Raphael Brodeur

    @Creation Date:     04/2025
    @Last modification: 04/2025

    @Description:       This file contains the script for testing the trained model for the MS-COCO experiments in the
                        paper.
"""

import matplotlib.pyplot as plt
from monai.metrics import DiceMetric
from monai.utils import set_determinism
import numpy as np
import torch
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

    # Split dataset into training and calibration sets (no offset in random from training.py so same split)
    train_ds, calib_ds, left_over_data = random_split(
        dataset=ds,
        lengths=[
            num_training_samples,                                       # 678
            num_calibration_samples,                                    # 20000
            len(ds) - num_training_samples - num_calibration_samples    # 43437
        ]
    )

    # Take 2869 samples from left_over_data
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

    # Metric
    dice_metric = DiceMetric()

    net.eval()
    dice_score_list = []
    non_conformity_score_list = []
    with torch.no_grad():
        for batch in test_loader:
            # Get image and ground truth seg
            x = batch.x.to(device)
            y = batch.y.to(device)

            # Get prediction
            y_pred = net(x)                     # Model forward pass
            y_pred = torch.sigmoid(y_pred)      # Sigmoid

            # Get Dice score
            dice_score = dice_metric(y=y, y_pred=torch.round(y_pred))     # Rounding the model prediction

            dice_score_list.append(dice_score.item())

            # Compare ground truth and prediction
            image = np.transpose(x[0].cpu().int().numpy(), (1, 2, 0))
            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(image)
            axes[0].imshow(y[0][0].cpu().numpy(), alpha=0.3)
            axes[0].set_title("Ground truth")
            axes[1].imshow(image)
            axes[1].imshow(y_pred[0][0].cpu().numpy(), alpha=0.3)
            axes[1].set_title("Model output")
            plt.show()

        # Stats on model performances
        print("Number of testing samples:", len(dice_score_list))
        print("Mean dice coefficient:", np.mean(dice_score_list))
        print("Std of dice coefficient:", np.std(dice_score_list))
