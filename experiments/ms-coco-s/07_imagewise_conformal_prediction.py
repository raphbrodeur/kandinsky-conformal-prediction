"""
    @file:              07_imagewise_conformal_prediction.py
    @Author:            Raphael Brodeur

    @Creation Date:     04/2025
    @Last modification: 04/2025

    @Description:       This file contains the script to perform image-wise conformal prediction for the MS-COCO-S
                        experiment from the paper.
"""

import matplotlib.pyplot as plt
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
    net.load_state_dict(torch.load("../ms-coco-xl/saved_params/model_params.pt", map_location=device))

    # Load calibration non_conformity_scores
    calib_non_conformity_scores = torch.load("../ms-coco-xl/saved_non_conformity_scores/non_conformity_scores.pt")

    # Use only 100 calibration samples
    calib_non_conformity_scores = calib_non_conformity_scores[:100]

    # Get image-wise non-conformity curves
    # Flatten H & W dims before computing the quantiles. Has shape (num_calib_samples, num_channels, H*W)
    calib_flatten_non_conformity_scores = torch.flatten(calib_non_conformity_scores, start_dim=2).cpu().numpy()

    # Get curve for all aggregated pixels in calibration set (aggregated from all images)
    non_conformity_curve = np.quantile(
        calib_flatten_non_conformity_scores,
        np.linspace(0, 1, 101),
        method="higher"              # Corresponds to ceil((n+1)(1-a)) / n quantile
    )

    non_conformity_curve = torch.from_numpy(non_conformity_curve).unsqueeze(-1).unsqueeze(-1)

    # Put back in image shape
    non_conformity_curves = non_conformity_curve.expand(101, *calib_non_conformity_scores.shape[2:]).unsqueeze(1)

    # Save non-conformity curves
    # torch.save(non_conformity_curves, "./saved_non_conformity_curves/imagewise_non_conformity_curves.pt")
    # print("Saved image-wise non-conformity curves.")


    # Examples

    # Get q_hat
    alpha = 0.25     # If alpha=0.1 then we want prob bound of 0.9; then we want quantile 0.9
    q_hat = non_conformity_curves[int((1 - alpha) * 100)].to(device)

    net.eval()
    with torch.no_grad():
        for batch in test_loader:
            # Get image and ground truth segmentation
            x = batch.x.to(device)
            y = batch.y.to(device)

            # Get model prediction
            y_pred = net(x)
            y_pred = torch.sigmoid(y_pred)

            # Get pixels for class label "segmentation" are in the prediction set
            # NOTE: this set contains all pixels for which class_label=1 is in the prediction set
            # (but class_label=0 may also be in there)
            y_pred_conformal = torch.where(y_pred >= (1 - q_hat), 1., 0.)   # 1 if pixel is in pred set, 0 otherwise

            # Post-processing
            y_pred = torch.round(y_pred)    # Rounding as a classification threshold


            # Compare ground truth, prediction and conformal prediction
            image = x[0][0].cpu().numpy()
            fig, axes = plt.subplots(1, 3)

            axes[0].imshow(image)
            axes[0].imshow(y[0][0].cpu().numpy(), alpha=0.3)
            axes[0].set_title("Ground truth")

            axes[1].imshow(image)
            axes[1].imshow(y_pred[0][0].cpu().numpy(), alpha=0.3)
            axes[1].set_title("Model output")

            axes[2].imshow(image)
            axes[2].imshow(y_pred_conformal[0][0].cpu().numpy(), alpha=0.3)
            axes[2].set_title("Conformal prediction set")
            plt.show()
