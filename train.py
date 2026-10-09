"""
train.py
--------
Regression-only training with the SCALE-MATCHED loss from the final
consultation:

    L = MSE(v_pred, v_true) + MSE(alpha_pred * r, alpha_true * r)

No weighting factor. Rescaling alpha by the radius puts both terms in
the same units (alpha*r = d - r, a physical distance), so — in the
lecturer's words — "you can get completely rid of W and nobody will
question the choice."

The model still PREDICTS the normalized alpha = d/r - 1 (bounded,
scale-free target); only the LOSS compares in distance units.

Classification stays evaluation-only: inside/outside from sign(alpha).
"""

from model import build_model
from dataset import CircleRegressionDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import time

import matplotlib
matplotlib.use("Agg")


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )


def combined_loss(v_pred, v_true, alpha_pred, alpha_true, radius, mse_loss):
    """Scale-matched loss: alpha rescaled by the per-sample radius so
    both terms live in coordinate units. radius: (B,) -> (B, 1)."""
    r = radius.unsqueeze(-1)
    vec_loss = mse_loss(v_pred, v_true)
    alpha_loss = mse_loss(alpha_pred * r, alpha_true * r)
    return vec_loss + alpha_loss, vec_loss, alpha_loss


@torch.no_grad()
def validate(model, val_loader, device, mse_loss):
    model.eval()
    total, total_vec, total_alpha, n = 0.0, 0.0, 0.0, 0
    for z, v_true, alpha_true, radius in val_loader:
        z = z.to(device)
        v_true = v_true.to(device)
        alpha_true = alpha_true.to(device)
        radius = radius.to(device)
        v_pred, alpha_pred = model(z)
        loss, vec_loss, alpha_loss = combined_loss(
            v_pred, v_true, alpha_pred, alpha_true, radius, mse_loss)
        total += loss.item()
        total_vec += vec_loss.item()
        total_alpha += alpha_loss.item()
        n += 1
    model.train()
    return total / n, total_vec / n, total_alpha / n


@torch.no_grad()
def evaluate(model, val_loader, val_dataset, device):
    """Returns (accuracy %, center error, runtime ms/sample).
    Accuracy from sign(alpha) — evaluation only, never in training."""
    model.eval()
    total_correct, total_points = 0, 0
    total_center_error, valid_center_count = 0.0, 0

    start = time.time()
    for z, v_true, alpha_true, _radius in val_loader:
        z = z.to(device)
        v_true = v_true.to(device)
        alpha_true = alpha_true.to(device)
        v_pred, alpha_pred = model(z)

        pred_inside = (alpha_pred < 0).float()
        true_inside = (alpha_true < 0).float()
        total_correct += (pred_inside == true_inside).sum().item()
        total_points += true_inside.numel()

        for i in range(z.size(0)):
            if true_inside[i].sum() > 0:
                xy = z[i, :, :2]
                true_center = (xy + v_true[i])[true_inside[i] == 1].mean(dim=0)
                mask = pred_inside[i] == 1
                if mask.sum() > 0:
                    pred_center = (xy + v_pred[i])[mask].mean(dim=0)
                    total_center_error += torch.norm(
                        pred_center - true_center).item()
                    valid_center_count += 1
    runtime_ms = (time.time() - start) / len(val_dataset) * 1000

    accuracy = total_correct / total_points * 100
    avg_center_error = total_center_error / max(1, valid_center_count)
    return accuracy, avg_center_error, runtime_ms


def train_and_evaluate(
    num_epochs=100,
    batch_size=32,
    num_points=100,
    noise_std=0.0,
    train_samples=5000,
    embed_dim=128,
    num_heads=1,
    num_layers=1,
    lr=1e-3,
    model_type="standard",
    early_stopping_patience=None,
    verbose=True,
    save_model=False,
    save_loss_curve=False,
):
    device = get_device()
    if verbose:
        print(f"[REGRESSION-ONLY, scale-matched loss] {model_type} | "
              f"points={num_points}, noise={noise_std}, dim={embed_dim}, "
              f"heads={num_heads}, layers={num_layers}, "
              f"samples={train_samples}, epochs={num_epochs} | {device}")

    train_dataset = CircleRegressionDataset(
        num_samples=train_samples, num_points=num_points, noise_std=noise_std)
    val_dataset = CircleRegressionDataset(
        num_samples=500, num_points=num_points, noise_std=noise_std, seed=42)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False)

    model = build_model(model_type, input_dim=3, embed_dim=embed_dim,
                        num_heads=num_heads, num_layers=num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val, best_state, patience_ctr = float("inf"), None, 0

    model.train()
    for epoch in range(num_epochs):
        epoch_loss, n_batches = 0.0, 0
        for z, v_true, alpha_true, radius in train_loader:
            z = z.to(device)
            v_true = v_true.to(device)
            alpha_true = alpha_true.to(device)
            radius = radius.to(device)

            optimizer.zero_grad()
            v_pred, alpha_pred = model(z)
            # Scale-matched loss — no weighting factor (consultation)
            loss, _, _ = combined_loss(
                v_pred, v_true, alpha_pred, alpha_true, radius, mse_loss)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_losses.append(epoch_loss / n_batches)
        val_loss, val_vec, val_alpha = validate(
            model, val_loader, device, mse_loss)
        val_losses.append(val_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{num_epochs}] "
                  f"train={train_losses[-1]:.4f}  val={val_loss:.4f} "
                  f"(vec={val_vec:.4f}, alpha*r={val_alpha:.4f})")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: t.detach().cpu().clone()
                          for k, t in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if (early_stopping_patience is not None
                    and patience_ctr >= early_stopping_patience):
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1} "
                          f"(best val={best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if save_loss_curve:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses,
                 label="Train Loss", linewidth=1.5)
        plt.plot(range(1, len(val_losses) + 1), val_losses,
                 label="Validation Loss", linewidth=1.5)
        plt.title("Regression-Only Model — Train / Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("loss_curve.png", dpi=150)
        plt.close()
        if verbose:
            print("  Loss curve saved to loss_curve.png")

    if save_model:
        torch.save(model.state_dict(), "model.pth")
        if verbose:
            print("  Model saved to model.pth")

    accuracy, avg_center_error, runtime_ms = evaluate(
        model, val_loader, val_dataset, device)

    if verbose:
        print(f"  -> Accuracy (sign of alpha, eval-only): {accuracy:.2f}%  |  "
              f"Center Error: {avg_center_error:.4f}  |  "
              f"Runtime: {runtime_ms:.2f} ms/sample")

    return accuracy, avg_center_error, runtime_ms


if __name__ == "__main__":
    train_and_evaluate(
        num_epochs=100,
        train_samples=5000,
        early_stopping_patience=15,
        save_model=True,
        save_loss_curve=True,
        verbose=True,
    )
