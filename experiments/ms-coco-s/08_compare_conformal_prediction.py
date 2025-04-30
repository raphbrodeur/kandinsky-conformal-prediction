"""
    @file:              08_compare_conformal_prediction.py
    @Author:            Raphael Brodeur

    @Creation Date:     04/2025
    @Last modification: 04/2025

    @Description:       This file contains the script to perform image-wise conformal prediction for the MS-COCO-S
                        experiment from the paper.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from monai.utils import set_determinism
import seaborn as sns
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
    alpha = 0.05

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

    # Load non-conformity curves
    pixelwise_nc_curves = torch.load("./saved_non_conformity_curves/pixelwise_non_conformity_curves.pt", map_location=device)
    imagewise_nc_curves = torch.load("./saved_non_conformity_curves/imagewise_non_conformity_curves.pt", map_location=device)
    kmeans_nc_curves = torch.load("./saved_non_conformity_curves/kmeans_non_conformity_curves.pt", map_location=device)
    genann_nc_curves = torch.load("./saved_non_conformity_curves/gen_ann_non_conformity_curves.pt", map_location=device)
    fcc_nc_curves = torch.load("./saved_non_conformity_curves/fcc_non_conformity_curves.pt", map_location=device)

    # Evaluate coverages
    net.eval()
    with torch.no_grad():
        # Get q_hat for each method for alpha factor defined above
        pixelwise_q_hat = pixelwise_nc_curves[int((1 - alpha) * 100)].to(device)
        imagewise_q_hat = imagewise_nc_curves[int((1 - alpha) * 100)].to(device)
        kmeans_q_hat = kmeans_nc_curves[int((1 - alpha) * 100)].to(device)
        genann_q_hat = genann_nc_curves[int((1 - alpha) * 100)].to(device)
        fcc_q_hat = fcc_nc_curves[int((1 - alpha) * 100)].to(device)

        pixelwise_coverages = []
        imagewise_coverages = []
        kmeans_coverages = []
        genann_coverages = []
        fcc_coverages = []
        for batch in test_loader:
            # Get image and ground truth segmentation
            x = batch.x.to(device)
            y = batch.y.to(device)

            # Get model prediction
            y_pred = net(x)
            y_pred = torch.sigmoid(y_pred)

            # Coverage for image (if and only if s(x,y)<=q_hat; so we get coverage easily by checking if s(x,y)<=q_hat)
            # (easier than checking if true label belongs to prediction set)
            test_non_conformity_scores = torch.where(y == 1, 1 - y_pred, y_pred)    # y=1 => s=1-f(x), y=0 => s=f(x)
            pixelwise_coverages.append((test_non_conformity_scores <= pixelwise_q_hat).float().mean().item())
            imagewise_coverages.append((test_non_conformity_scores <= imagewise_q_hat).float().mean().item())
            kmeans_coverages.append((test_non_conformity_scores <= kmeans_q_hat).float().mean().item())
            genann_coverages.append((test_non_conformity_scores <= genann_q_hat).float().mean().item())
            fcc_coverages.append((test_non_conformity_scores <= fcc_q_hat).float().mean().item())

        print(f"Pixelwise: {1 - np.mean(pixelwise_coverages)} [{1-np.quantile(pixelwise_coverages, 0.95)}, {1 - np.quantile(pixelwise_coverages, 0.05)}]")
        print(f"Imagewise: {1 - np.mean(imagewise_coverages)} [{1-np.quantile(imagewise_coverages, 0.95)}, {1 - np.quantile(imagewise_coverages, 0.05)}]")
        print(f"KMeans: {1 - np.mean(kmeans_coverages)} [{1-np.quantile(kmeans_coverages, 0.95)}, {1 - np.quantile(kmeans_coverages, 0.05)}]")
        print(f"GenAnn: {1 - np.mean(genann_coverages)} [{1-np.quantile(genann_coverages, 0.95)}, {1 - np.quantile(genann_coverages, 0.05)}]")
        print(f"FCC: {1 - np.mean(fcc_coverages)} [{1-np.quantile(fcc_coverages, 0.95)}, {1 - np.quantile(fcc_coverages, 0.05)}]")

        # Violin plot
        d = {
            "Pixelwise": 1 - np.array(pixelwise_coverages),
            "Imagewise": 1 - np.array(imagewise_coverages),
            "KMeans": 1 - np.array(kmeans_coverages),
            "GenAnn": 1 - np.array(genann_coverages),
            "FCC": 1 - np.array(fcc_coverages)
        }
        df = pd.DataFrame(data=d)
        sns.violinplot(data=df)
        plt.title(f"Conformal prediction coverage error distribution alpha={alpha}")
        plt.ylabel("Coverage error")
        plt.show()