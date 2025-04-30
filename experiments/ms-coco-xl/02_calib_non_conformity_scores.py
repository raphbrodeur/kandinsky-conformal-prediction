"""
    @file:              02_calib_non_conformity_scores.py
    @Author:            Raphael Brodeur

    @Creation Date:     04/2025
    @Last modification: 04/2025

    @Description:       This file contains the script to compute the non-conformity scores for the MS-COCO-XL experiment
                        from the paper.
"""

from monai.utils import set_determinism
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
    train_ds, calib_ds, _ = random_split(
        dataset=ds,
        lengths=[
            num_training_samples,                                       # 678
            num_calibration_samples,                                    # 20000
            len(ds) - num_training_samples - num_calibration_samples    # 43437
        ]
    )

    calib_loader = DataLoader(
        dataset=calib_ds,
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

    net.eval()
    non_conformity_score_list = []
    with torch.no_grad():
        for batch in calib_loader:
            # Get image and ground truth segmentation
            x = batch.x.to(device)
            y = batch.y.to(device)

            # Get model prediction
            y_pred = net(x)
            y_pred = torch.sigmoid(y_pred)

            # Get non-conformity score (1 - soft prob of ground truth class output by model for each pixel)
            non_conformity_score = torch.where(y == 1, 1 - y_pred, y_pred)  # Since y_pred is not one-hot encoded

            non_conformity_score_list.append(non_conformity_score.cpu())

        # Concatenate non-conformity scores from all samples in calibration set
        calib_non_conformity_scores = torch.cat(non_conformity_score_list, dim=0)

        # Save calibration non-conformity scores
        # torch.save(calib_non_conformity_scores, "./saved_non_conformity_scores/non_conformity_scores.pt")
        # print("Saved calibration set non-conformity scores.")
