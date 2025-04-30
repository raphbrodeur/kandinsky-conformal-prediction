"""
    @file:              04_kmeans_conformal_prediction.py
    @Author:            Raphael Brodeur

    @Creation Date:     04/2025
    @Last modification: 04/2025

    @Description:       This file contains the script to perform k-means kandinsky conformal prediction for the
                        Decathlon-L experiment from the paper.
"""

import matplotlib.pyplot as plt
from monai.utils import set_determinism
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.utils.data import DataLoader, random_split

from src.data import DecathlonDataset, SlicedDecathlonDataset
from src.models import UNetPaper


if __name__ == "__main__":
    # Set random seed
    set_determinism(seed=1010710)

    # Hyperparameters
    num_training_samples = 86
    num_calibration_samples = 77
    num_clusters = 4                # k = 4 in code from paper

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_workers = 0

    # Dataset
    ds = DecathlonDataset(path_to_dir="C:/Users/Labo/Desktop/datasets/decathlon/Task07_Pancreas")

    # Split dataset into training and calibration sets (no offset in random from training.py so same split)
    train_ds, calib_ds, left_over_data = random_split(
        dataset=ds,
        lengths=[
            num_training_samples,                                       # 86
            num_calibration_samples,                                    # 77
            len(ds) - num_training_samples - num_calibration_samples    # 118
        ]
    )

    # Slices dataset
    test_ds = SlicedDecathlonDataset(
        dataset=train_ds,
        size=[384, 384],
        apply_augmentations=False
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
        in_channels=1,
        out_channels=1,
        channels=128
    ).to(device)

    # Load model weights
    net.load_state_dict(torch.load("../decathlon-l/saved_params/model_params.pt", map_location=device))

    # Load calibration non-conformity_scores
    calib_non_conformity_scores = torch.load("../decathlon-l/saved_non_conformity_scores/non_conformity_scores.pt")

    # Use only 27 calibration samples
    calib_non_conformity_scores = calib_non_conformity_scores[:27]

    # Get pixel-wise non-conformity curves for clustering
    non_conformity_curves = torch.quantile(
        calib_non_conformity_scores,
        torch.linspace(0, 1, 101),          # For each pixel, get a curve of q_hat for 1-alpha=0.0,...,1.0
        dim=0,                              # Pixel-wise
        interpolation="higher"              # Corresponds to ceil((n+1)(1-a)) / n quantile
    )

    # Cluster pixels based on similarity of their non-conformity curves (k-means clustering)
    kmeans_cluster_finder = KMeans(
        n_clusters=num_clusters,
        random_state=0
    )

    # Approximate non-conformity curves are used in the paper's code to determine the clusters but no mention in paper.
    ...

    # Get mask of cluster labels for each pixel. NOTE: need .permute() so .reshape() (row-major order) works properly !
    kandinsky_mask = kmeans_cluster_finder.fit_predict(
        non_conformity_curves.permute(1, 2, 3, 0)[0].reshape(-1, non_conformity_curves.shape[0]).cpu().numpy()
    )

    # Reshape kandinsky mask to original 2d image shape
    kandinsky_mask = kandinsky_mask.reshape(non_conformity_curves.shape[2], non_conformity_curves.shape[3])

    # Plot the kandinsky mask
    plt.imshow(kandinsky_mask, aspect='auto', origin='upper')
    plt.colorbar()  # Add a colorbar to visualize cluster labels
    plt.title("Kandinsky Mask Decathlon-L (K-Means)")
    plt.show()

    # Get a non-conformity curve for each cluster
    for cluster_label in range(num_clusters):
        # Aggregate all non-conformity scores of every pixel in the calibration set belonging to the cluster
        cluster_non_conformity_scores = calib_non_conformity_scores[:, 0, kandinsky_mask == cluster_label].flatten().numpy()

        # Get a non-conformity curve for the cluster
        cluster_non_conformity_curve = np.quantile(
            cluster_non_conformity_scores,
            np.linspace(0, 1, 101), # For each pixel, get a curve of q_hat for 1-alpha=0.0,...,1.0
            method="higher"         # Corresponds to ceil((n+1)(1-a)) / n quantile
        )

        # Create tensor with cluster curve at every position and reshape to shape (101, H, W)
        cluster_non_conformity_curve = torch.from_numpy(
            cluster_non_conformity_curve
        ).unsqueeze(-1).unsqueeze(-1).expand(101, *calib_non_conformity_scores.shape[2:])

        # Update non-conformity curves of pixels belonging to cluster
        non_conformity_curves[:, 0, kandinsky_mask == cluster_label] = cluster_non_conformity_curve[:, kandinsky_mask == cluster_label]

    # Save non-conformity curves
    # torch.save(non_conformity_curves, "./saved_non_conformity_curves/kmeans_non_conformity_curves.pt")
    # print("Saved k-means non-conformity curves.")


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
