"""
dataset.py
----------
Synthetic circle point-cloud dataset — REGRESSION-ONLY version.

Each sample is a point cloud of |S| points in [-1, 1]^2 with a grayscale
intensity channel. A circle of random radius/position colors points
inside it differently from points outside.

Targets (both regression, per lecturer's final instruction):
    v      : (N, 2)  displacement vector from each point to the circle center
    alpha  : (N,)    normalized signed distance to the circle boundary
                       alpha = dist(point, center) / radius - 1
                       alpha < 0  -> inside,  alpha > 0 -> outside
    radius : ()      circle radius r — returned so the loss can rescale
                     alpha back to distance units (alpha*r = d - r),
                     putting both loss terms on the same scale with no
                     tunable weight (lecturer's suggestion).

There is NO binary classification label. Inside/outside is inferred
from sign(alpha) at evaluation time only.

Noise (updated per consultation): Gaussian noise is applied to the
POINT COORDINATES ONLY. Grayscale values are left exact — adding noise
to intensities pushes them outside [0, 1] (or clips), creating values
never seen in training, which confounds the robustness measurement.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class CircleRegressionDataset(Dataset):
    def __init__(
        self,
        num_samples,
        num_points=100,
        noise_std=0.0,
        add_clutter=False,
        inside_color=1.0,
        outside_color=0.0,
        seed=None,
    ):
        """
        Args:
            num_samples:   number of point clouds in the dataset
            num_points:    points per cloud |S|
            noise_std:     Gaussian noise std applied per point to the
                           xy coordinates ONLY (grayscale stays exact)
            add_clutter:   optionally paint a random rectangle with a
                           random intensity (distractor object)
            inside_color:  intensity of points inside the circle
            outside_color: intensity of points outside the circle
            seed:          if set, sample idx is generated with seed+idx
                           (deterministic — use for validation sets);
                           if None, every access is freshly random
        """
        self.num_samples = num_samples
        self.num_points = num_points
        self.noise_std = noise_std
        self.add_clutter = add_clutter
        self.inside_color = inside_color
        self.outside_color = outside_color
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rng = np.random.default_rng(
            None if self.seed is None else self.seed + idx
        )

        # Circle fully inside the patch
        radius = rng.uniform(0.2, 0.6)
        cx = rng.uniform(-1.0 + radius, 1.0 - radius)
        cy = rng.uniform(-1.0 + radius, 1.0 - radius)

        points_xy = rng.uniform(-1.0, 1.0, size=(self.num_points, 2))

        dist = np.sqrt((points_xy[:, 0] - cx) ** 2 +
                       (points_xy[:, 1] - cy) ** 2)
        inside = dist <= radius
        colors = np.where(inside, self.inside_color, self.outside_color)

        # Per-point noise on COORDINATES ONLY (consultation change):
        # grayscale values stay exactly inside_color / outside_color
        if self.noise_std > 0.0:
            points_xy = points_xy + rng.normal(
                0.0, self.noise_std, size=(self.num_points, 2))

        if self.add_clutter and rng.random() > 0.5:
            rect_x, rect_y = rng.uniform(-0.8, 0.8, 2)
            rect_w, rect_h = rng.uniform(0.1, 0.4, 2)
            in_rect = (
                (points_xy[:, 0] >= rect_x) & (points_xy[:, 0] <= rect_x + rect_w) &
                (points_xy[:, 1] >= rect_y) & (
                    points_xy[:, 1] <= rect_y + rect_h)
            )
            colors = colors.astype(np.float64)
            colors[in_rect] = rng.uniform(0.0, 1.0)

        # Regression targets from the (noisy) observed positions to the
        # ORIGINAL circle — the ground-truth circle does not move
        v = np.stack([cx - points_xy[:, 0], cy - points_xy[:, 1]], axis=-1)
        dist_now = np.sqrt((points_xy[:, 0] - cx) ** 2 +
                           (points_xy[:, 1] - cy) ** 2)
        alpha = dist_now / (radius + 1e-8) - 1.0

        z = np.concatenate(
            [points_xy, colors[:, None]], axis=-1).astype(np.float32)

        return (
            torch.from_numpy(z),
            torch.from_numpy(v.astype(np.float32)),
            torch.from_numpy(alpha.astype(np.float32)),
            torch.tensor(radius, dtype=torch.float32),
        )
