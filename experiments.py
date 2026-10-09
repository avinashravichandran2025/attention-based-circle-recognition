"""
experiments.py
--------------
All experiments, rerun with the REGRESSION-ONLY model.

  Exp 1: number of points |S| (accuracy + runtime log-log vs O(N^2))
  Exp 2: noise sweep 0.0 -> 1.0
  Exp 3: model complexity (dim / heads / layers)
  Exp 4: training set size
  Exp 5: standard embedding vs no-embedding first layer
  Exp 6: vector-loss weighting sweep (how the balance between the
         vector MSE and alpha MSE affects results — discussed in the
         consultation)

Each experiment saves a CSV and a PNG in the working directory.
"""

from train import train_and_evaluate
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


EPOCHS = 100
TRAIN_SIZE = 5000


def run_experiment_1():
    print("\n--- Experiment 1: Varying Number of Points |S| ---")
    points_list = [10, 25, 50, 100, 200]
    results = []
    for pts in points_list:
        acc, err, rt = train_and_evaluate(
            num_epochs=EPOCHS, num_points=pts,
            train_samples=TRAIN_SIZE, verbose=True)
        results.append({"Points": pts, "Accuracy": acc,
                        "Error": err, "Runtime(ms)": rt})

    df = pd.DataFrame(results)
    print(df)
    df.to_csv("exp1_points.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(df["Points"], df["Accuracy"], marker="o", linewidth=2)
    axes[0].set_title("Exp 1: Accuracy vs Number of Points")
    axes[0].set_xlabel("Number of Points |S|")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["Points"], df["Error"], marker="o",
                 linewidth=2, color="coral")
    axes[1].set_title("Exp 1: Center Error vs Number of Points")
    axes[1].set_xlabel("Number of Points |S|")
    axes[1].set_ylabel("Center Prediction Error (L2)")
    axes[1].grid(True, alpha=0.3)

    axes[2].loglog(df["Points"], df["Runtime(ms)"], marker="o",
                   linewidth=2, label="Measured")
    n = df["Points"].values
    ref = n ** 2 / n[0] ** 2 * df["Runtime(ms)"].values[0]
    axes[2].loglog(n, ref, "--", color="gray", label="O(N²) reference")
    axes[2].set_title("Exp 1: Runtime vs Points (log-log)")
    axes[2].set_xlabel("Number of Points |S|")
    axes[2].set_ylabel("Runtime (ms/sample)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig("exp1_points.png", dpi=150)
    plt.close()
    print("Saved exp1_points.png")


def run_experiment_2():
    print("\n--- Experiment 2: Varying Noise Level ---")
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    results = []
    for noise in noise_levels:
        acc, err, _ = train_and_evaluate(
            num_epochs=EPOCHS, noise_std=noise,
            train_samples=TRAIN_SIZE, verbose=True)
        results.append({"Noise": noise, "Accuracy": acc, "Error": err})

    df = pd.DataFrame(results)
    print(df)
    df.to_csv("exp2_noise.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(df["Noise"], df["Accuracy"], marker="o",
                 linewidth=2, color="steelblue")
    axes[0].set_title("Exp 2: Accuracy vs Noise Level")
    axes[0].set_xlabel("Gaussian Noise Std Dev")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["Noise"], df["Error"], marker="o",
                 linewidth=2, color="coral")
    axes[1].set_title("Exp 2: Center Error vs Noise Level")
    axes[1].set_xlabel("Gaussian Noise Std Dev")
    axes[1].set_ylabel("Center Prediction Error (L2)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("exp2_noise.png", dpi=150)
    plt.close()
    print("Saved exp2_noise.png")


def run_experiment_3():
    print("\n--- Experiment 3: Varying Model Complexity ---")
    configs = [
        {"dim": 64,  "heads": 1, "layers": 1},
        {"dim": 128, "heads": 1, "layers": 1},
        {"dim": 128, "heads": 2, "layers": 1},
        {"dim": 128, "heads": 1, "layers": 2},
        {"dim": 128, "heads": 4, "layers": 3},
    ]
    results = []
    for conf in configs:
        acc, err, rt = train_and_evaluate(
            num_epochs=EPOCHS, embed_dim=conf["dim"],
            num_heads=conf["heads"], num_layers=conf["layers"],
            train_samples=TRAIN_SIZE, verbose=True)
        results.append({
            "Config": f"{conf['dim']}d_{conf['heads']}h_{conf['layers']}l",
            "Accuracy": acc, "Error": err, "Runtime(ms)": rt})

    df = pd.DataFrame(results)
    print(df)
    df.to_csv("exp3_complexity.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = range(len(df))
    axes[0].bar(x, df["Accuracy"], color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["Config"], rotation=15)
    axes[0].set_title("Exp 3: Accuracy by Model Complexity")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 105)
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(x, df["Error"], color="coral")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["Config"], rotation=15)
    axes[1].set_title("Exp 3: Center Error by Model Complexity")
    axes[1].set_ylabel("Center Prediction Error (L2)")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("exp3_complexity.png", dpi=150)
    plt.close()
    print("Saved exp3_complexity.png")


def run_experiment_4():
    print("\n--- Experiment 4: Varying Training Set Size ---")
    sizes = [1000, 5000, 10000]
    results = []
    for size in sizes:
        acc, err, _ = train_and_evaluate(
            num_epochs=EPOCHS, train_samples=size, verbose=True)
        results.append({"Train Size": size, "Accuracy": acc, "Error": err})

    df = pd.DataFrame(results)
    print(df)
    df.to_csv("exp4_trainsize.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(df["Train Size"], df["Accuracy"], marker="o",
                 linewidth=2, color="steelblue")
    axes[0].set_title("Exp 4: Accuracy vs Training Set Size")
    axes[0].set_xlabel("Number of Training Samples")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["Train Size"], df["Error"], marker="o",
                 linewidth=2, color="coral")
    axes[1].set_title("Exp 4: Center Error vs Training Set Size")
    axes[1].set_xlabel("Number of Training Samples")
    axes[1].set_ylabel("Center Prediction Error (L2)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("exp4_trainsize.png", dpi=150)
    plt.close()
    print("Saved exp4_trainsize.png")


def run_experiment_5():
    print("\n--- Experiment 5: Standard vs No-Embedding Transformer ---")
    results = []
    for m in ["standard", "no_embedding"]:
        acc, err, rt = train_and_evaluate(
            num_epochs=EPOCHS, train_samples=TRAIN_SIZE,
            model_type=m, verbose=True)
        results.append({"Model Type": m, "Accuracy": acc,
                        "Error": err, "Runtime(ms)": rt})

    df = pd.DataFrame(results)
    print(df)
    df.to_csv("exp5_embedding_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = range(len(df))
    axes[0].bar(x, df["Accuracy"], color=["steelblue", "coral"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["Model Type"])
    axes[0].set_title("Exp 5: Accuracy — Standard vs No-Embedding")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 105)
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(x, df["Error"], color=["steelblue", "coral"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["Model Type"])
    axes[1].set_title("Exp 5: Center Error — Standard vs No-Embedding")
    axes[1].set_ylabel("Center Prediction Error (L2)")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("exp5_embedding_comparison.png", dpi=150)
    plt.close()
    print("Saved exp5_embedding_comparison.png")


if __name__ == "__main__":
    print("Starting all experiments (regression-only model)...")
    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
    run_experiment_4()
    run_experiment_5()
    print("\nAll experiments complete.")
