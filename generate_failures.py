"""
generate_failures.py
--------------------
Qualitative failure cases for the regression-only model.
Fixed seeds so the plots are identical every run (good for slides).

Scenarios:
  1. High noise (std = 1.0)
  2. Low contrast (inside/outside intensities 0.50 vs 0.45)
  3. Few points (|S| = 10)
"""

from train import get_device
from model import build_model
from dataset import CircleRegressionDataset
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


EMBED_DIM = 128
NUM_HEADS = 1
NUM_LAYERS = 1


def main():
    print("Generating qualitative failure cases...")
    device = get_device()

    model = build_model("standard", input_dim=3, embed_dim=EMBED_DIM,
                        num_heads=NUM_HEADS, num_layers=NUM_LAYERS).to(device)
    try:
        model.load_state_dict(
            torch.load("model.pth", map_location=device, weights_only=True))
        print("Loaded trained weights from model.pth")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Warning: {e}")
        print("Using untrained model — run train.py first.")
    model.eval()

    scenarios = [
        ("High Noise (std=1.0)",
         CircleRegressionDataset(1, num_points=100, noise_std=1.0, seed=10)[0]),
        ("Low Contrast (Δ=0.05)",
         CircleRegressionDataset(1, num_points=100, noise_std=0.05,
                                 inside_color=0.5, outside_color=0.45,
                                 seed=20)[0]),
        ("Few Points (|S|=10)",
         CircleRegressionDataset(1, num_points=10, noise_std=0.1, seed=30)[0]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    with torch.no_grad():
        for ax, (title, (z, v_true, alpha_true, _radius)) in zip(axes, scenarios):
            v_pred_t, alpha_pred_t = model(z.unsqueeze(0).to(device))
            v_pred = v_pred_t.squeeze(0).cpu().numpy()
            alpha_pred = alpha_pred_t.squeeze(0).cpu().numpy()

            z_np, v_true_np = z.numpy(), v_true.numpy()
            alpha_true_np = alpha_true.numpy()
            xy = z_np[:, :2]

            true_inside = alpha_true_np < 0
            pred_inside = alpha_pred < 0

            # Points colored by predicted alpha sign (blue in, red out)
            ax.scatter(xy[:, 0], xy[:, 1],
                       c=np.where(pred_inside, 1.0, 0.0),
                       cmap="coolwarm_r", vmin=0, vmax=1, s=30, zorder=2)

            ax.quiver(xy[:, 0], xy[:, 1], v_pred[:, 0], v_pred[:, 1],
                      angles="xy", scale_units="xy", scale=1,
                      color="salmon", alpha=0.6, label="Predicted Vector")

            if np.any(true_inside):
                true_center = (xy + v_true_np)[true_inside].mean(axis=0)
                ax.scatter(*true_center, color="blue", marker="*",
                           s=200, zorder=3, label="True Center")
            if np.any(pred_inside):
                pred_center = (xy + v_pred)[pred_inside].mean(axis=0)
                ax.scatter(*pred_center, color="green", marker="X",
                           s=200, zorder=3, label="Pred Center")

            ax.set_title(title, fontsize=13)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect("equal", adjustable="box")
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("failure_cases.png", dpi=150)
    plt.close()
    print("Saved failure_cases.png")


if __name__ == "__main__":
    main()
