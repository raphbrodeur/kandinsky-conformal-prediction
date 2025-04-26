"""
    @file:              conformal_calibration.py
    @Author:            Raphael Brodeur

    @Creation Date:     04/2025
    @Last modification: 04/2025

    @Description:       This file contains the handling of the various split conformal prediction calibration methods in
                        the paper.
"""

import torch
from torch.nn import Module
from torch.utils.data import DataLoader


class ConformalPrediction:
    """
    This class handles various split conformal prediction calibration methods from the paper.
    """

    def __init__(
            self,
            model: Module,
            device: torch.device,
            calibration_data: DataLoader,
            mode: str
    ):
        """
        ...
        """
        self._model = model
        self._device = device
        self._calibration_data = calibration_data
        self._mode = mode

    def _get_non_conformity_scores(self) -> torch.Tensor:
        """
        Computes the non conformity score maps.
        """
        # Ensure model is in validation mode
        self._model.eval()

        with torch.no_grad():
            non_conformity_score_per_batch = []
            for batch in self._calibration_data:
                # Get image and ground truth segmentation
                x = batch.x.to(self._device)
                y = batch.y.to(self._device)

                # Get model prediction
                y_pred = self._model(x)
                y_pred = torch.sigmoid(y_pred)  # Pseudo probs: each pixel has value in [0, 1]

                # Get non conformity scores
                non_conformity_score = 1 - y_pred

                # Assign NaN values where y = 0 ...? TODO
                # non_conformity_score = torch.where(y == 1, non_conformity_score, torch.tensor(float("nan")))

                non_conformity_score_per_batch.append(non_conformity_score.cpu())

            non_conformity_scores = torch.cat(non_conformity_score_per_batch, dim=0)

            return non_conformity_scores

    def _get_non_conformity_curves(self):
        """
        ...
        """
        non_conformity_scores = self._get_non_conformity_scores()  # Has shape (num_cal, num_channels, H, W)

        # Pixel-wise calibration
        if self._mode == "pixel-wise":
            # Get NC curve for each pixel
            # Has shape (num_curve_points, num_channels, H, W). Curve for a pixel is along num_curve_points dimension.
            non_conformity_curve = torch.quantile(non_conformity_scores, torch.linspace(0, 1, 100), dim=0)

        # Image-wise calibration
        elif self._mode == "image-wise":
            # Flatten H & W dims before computing the quantiles. Has shape (num_cal, num_channels, H*W)
            non_conformity_scores_flat = torch.flatten(non_conformity_scores, start_dim=2)

            # Get NC curve for each image. Tensor of length num_curve_points. Curve is along num_curve_points_dim.
            non_conformity_curve = torch.quantile(non_conformity_scores_flat, torch.linspace(0, 1, 100))

        ...

        return non_conformity_curve




