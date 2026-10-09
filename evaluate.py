"""
evaluate.py
-----------
Loads the trained regression-only model (model.pth from train.py) and:
  1. Reports metrics on a 500-sample seeded evaluation set:
       - classification accuracy (inferred from sign of alpha, eval-only)
       - center prediction error (L2)
       - inference runtime per sample
  2. Saves a 4-panel qualitative figure (evaluation_results.png):
       Panel 1: input point cloud (grayscale intensities)
       Panel 2: predicted alpha (diverging colormap centered at 0 —
                blue inside / red outside, boundary at white)
       Panel 3: predicted displacement vectors + predicted/true center
       Panel 4: ground truth displacement vectors + true center
"""

from train import evaluate, get_device
from model import build_model
from dataset import CircleRegressionDataset
from torch.utils.data import DataLoader
from matplotlib.colors import TwoSlopeNorm
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


EMBED_DIM = 128
NUM_HEADS = 1
NUM_LAYERS = 1


def main():
    device = get_device()

    model = build_model("standard", input_dim=3, embed_dim=EMBED_DIM,
                        num_heads=NUM_HEADS, num_layers=NUM_LAYERS).to(device)
    model.load_state_dict(
        torch.load("model.pth", map_location=device, weights_only=True))
    model.eval()

    # Quantitative evaluation
    eval_dataset = CircleRegressionDataset(
        num_samples=500, num_points=100, noise_std=0.0, seed=42)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
    accuracy, center_error, runtime_ms = evaluate(
        model, eval_loader, eval_dataset, device)

    print(f"Evaluation over {len(eval_dataset)} patches:")
    print(f"  Classification Accuracy (sign of alpha): {accuracy:.2f}%")
    print(f"  Center Prediction Error (L2):            {center_error:.4f}")
    print(
        f"  Avg Inference Time:                      {runtime_ms:.2f} ms/sample")

    # Qualitative figure on one noisy sample with clutter
    vis_dataset = CircleRegressionDataset(
        num_samples=1, num_points=100, noise_std=0.1,
        add_clutter=True, seed=7)
    z, v_true, alpha_true, _radius = vis_dataset[0]

    with torch.no_grad():
        v_pred_t, alpha_pred_t = model(z.unsqueeze(0).to(device))
    v_pred = v_pred_t.squeeze(0).cpu().numpy()
    alpha_pred = alpha_pred_t.squeeze(0).cpu().numpy()

    z_np, v_true_np = z.numpy(), v_true.numpy()
    alpha_true_np = alpha_true.numpy()
    xy = z_np[:, :2]

    true_inside = alpha_true_np < 0
    pred_inside = alpha_pred < 0

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Panel 1 — input
    axes[0].scatter(xy[:, 0], xy[:, 1], c=z_np[:, 2],
                    cmap="gray", vmin=0, vmax=1, s=20)
    axes[0].set_facecolor("#d9d9d9")
    axes[0].set_title("Input Point Cloud (Grayscale)")

    # Panel 2 — predicted alpha, centered diverging colormap
    sc = axes[1].scatter(
        xy[:, 0], xy[:, 1], c=alpha_pred, cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0), s=20)
    axes[1].set_title("Predicted alpha (<0 inside, >0 outside)")
    plt.colorbar(sc, ax=axes[1])

    # Panel 3 — predicted vectors
    axes[2].scatter(xy[:, 0], xy[:, 1], color="gray", alpha=0.3, s=10)
    axes[2].quiver(xy[:, 0], xy[:, 1], v_pred[:, 0], v_pred[:, 1],
                   angles="xy", scale_units="xy", scale=1,
                   color="salmon", alpha=0.7, label="Predicted Vector")
    if np.any(true_inside):
        true_center = (xy + v_true_np)[true_inside].mean(axis=0)
        axes[2].scatter(*true_center, color="blue", marker="*",
                        s=200, zorder=3, label="True Center")
    if np.any(pred_inside):
        pred_center = (xy + v_pred)[pred_inside].mean(axis=0)
        axes[2].scatter(*pred_center, color="green", marker="X",
                        s=200, zorder=3, label="Pred Center")
    axes[2].set_title("Predicted Displacement Vectors")
    axes[2].legend(fontsize=8)

    # Panel 4 — ground truth vectors
    axes[3].scatter(xy[:, 0], xy[:, 1], color="gray", alpha=0.3, s=10)
    axes[3].quiver(xy[:, 0], xy[:, 1], v_true_np[:, 0], v_true_np[:, 1],
                   angles="xy", scale_units="xy", scale=1,
                   color="blue", alpha=0.7, label="True Vector")
    if np.any(true_inside):
        true_center = (xy + v_true_np)[true_inside].mean(axis=0)
        axes[3].scatter(*true_center, color="blue", marker="*",
                        s=200, zorder=3, label="True Center")
    axes[3].set_title("Ground Truth Displacement Vectors")
    axes[3].legend(fontsize=8)

    for ax in axes:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig("evaluation_results.png", dpi=150)
    plt.close()
    print("Evaluation plot saved to evaluation_results.png")


if __name__ == "__main__":
    main()
