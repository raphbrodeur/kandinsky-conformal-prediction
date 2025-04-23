"""
    @file:              training.py
    @Author:            Raphael Brodeur

    @Creation Date:     04/2025
    @Last modification: 04/2025

    @Description:       This file contains the training of the model for the MS-COCO experiments in the paper.
"""

from monai.losses import DiceLoss
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
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 35

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_workers = 0

    # Dataset
    ds = COCODataset(
        path_to_dir="C:/Users/Labo/Desktop/datasets/coco_2017_seg/learning",
        size=[320, 240],
        apply_augmentations=True
    )

    # Split dataset into training and calibration sets
    train_ds, calib_ds, _ = random_split(
        dataset=ds,
        lengths=[
            num_training_samples,
            num_calibration_samples,
            len(ds) - num_training_samples - num_calibration_samples
        ]
    )

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # Model
    net = UNetPaper(
        in_channels=3,
        out_channels=1,
        channels=128
    ).to(device)

    # Optimizer
    opt = torch.optim.AdamW(params=net.parameters(), lr=learning_rate)

    # Loss
    dice_loss = DiceLoss()

    # Training loop
    loss_per_epoch = []
    for epoch in range(num_epochs):
        # Training
        net.train()

        loss_per_batch = []
        for batch in train_loader:
            # Get image and ground truth seg
            x = batch.x.to(device)
            y = batch.y.to(device)

            # Reset grad
            opt.zero_grad()

            # Get prediction
            y_pred = net(x)                     # Model forward pass
            y_pred = torch.sigmoid(y_pred)      # Sigmoid

            # Get loss
            loss = dice_loss(input=y_pred, target=y)

            # Update params
            loss.backward()
            opt.step()

            loss_per_batch.append(loss.item())

        loss_per_epoch.append(np.mean(loss_per_batch))

        # No validation step...

        print(f"Epoch {epoch}, Dice loss: {loss_per_epoch[-1]}")

        # Save model parameters
        torch.save(net.state_dict(), "./saved_params/model_params.pt")
        print("Model parameters saved.")

    print("Training finished.")
    print(loss_per_epoch)
