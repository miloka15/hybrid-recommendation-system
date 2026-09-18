"""
File: 08b_evaluation_coldstart.py
Project: Hybrid Recommendation System

Purpose:
Loads the saved predictions and metrics from 07b_hybrid_model_coldstart.py
and produces the same diagnostic tables/figures as 08_evaluation.py, but
for the leakage-free (leave-one-out product aggregate) hybrid model, so
it can be compared directly against the same-review hybrid model.
"""

import pandas as pd
import numpy as np
import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configure import FIGURES_DIR

PREDICTIONS_PATH = "results/tables/hybrid_coldstart_predictions.csv"
METRICS_PATH = "results/metrics/hybrid_model_coldstart.json"
IMPORTANCE_PATH = "results/tables/hybrid_coldstart_feature_importances.csv"


def load_predictions():
    df = pd.read_csv(PREDICTIONS_PATH)
    print(f"Loaded cold-start predictions with shape: {df.shape}")
    return df


def load_metrics():
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    print("Loaded cold-start headline metrics:")
    print(json.dumps(metrics, indent=2))
    return metrics


def load_feature_importances():
    df = pd.read_csv(IMPORTANCE_PATH)
    print(f"Loaded feature importances with shape: {df.shape}")
    return df


def compute_metrics_by_rating(df):
    print("\n" + "*" * 50)
    print("COLD-START MODEL: METRICS BY RATING CLASS")
    print("*" * 50)

    rows = []
    for rating, group in df.groupby("Actual"):
        hybrid_rmse = np.sqrt(np.mean((group["Actual"] - group["Predicted"]) ** 2))
        hybrid_mae = np.mean(np.abs(group["Actual"] - group["Predicted"]))
        svd_rmse = np.sqrt(np.mean((group["Actual"] - group["SVD_Only_Predicted"]) ** 2))
        svd_mae = np.mean(np.abs(group["Actual"] - group["SVD_Only_Predicted"]))

        rows.append({
            "rating": rating, "n_samples": len(group),
            "coldstart_hybrid_rmse": hybrid_rmse, "coldstart_hybrid_mae": hybrid_mae,
            "svd_only_rmse": svd_rmse, "svd_only_mae": svd_mae,
        })

    by_rating_df = pd.DataFrame(rows)
    print(by_rating_df.to_string(index=False))
    return by_rating_df


def plot_overall_comparison(metrics, save_dir):
    fig, ax = plt.subplots(figsize=(6, 5))

    labels = ["RMSE", "MAE"]
    hybrid_vals = [metrics["hybrid_coldstart_rmse"], metrics["hybrid_coldstart_mae"]]
    svd_vals = [metrics["svd_baseline_rmse"], metrics["svd_baseline_mae"]]

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, svd_vals, width, label="SVD-Only Baseline", color="#a0a0a0")
    ax.bar(x + width / 2, hybrid_vals, width, label="Cold-Start Hybrid", color="#55A868")

    ax.set_ylabel("Error")
    ax.set_title("Cold-Start Hybrid vs. SVD-Only Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for i, (svd_v, hyb_v) in enumerate(zip(svd_vals, hybrid_vals)):
        ax.text(i - width / 2, svd_v + 0.01, f"{svd_v:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, hyb_v + 0.01, f"{hyb_v:.3f}", ha="center", fontsize=9)

    fig.tight_layout()
    path = os.path.join(save_dir, "coldstart_overall_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_leaky_vs_coldstart_comparison(save_dir):
    """
    Three-way bar chart: SVD-only vs. same-review (leaky) hybrid vs.
    leave-one-out (cold-start-safe) hybrid, for both RMSE and MAE.
    Reads both metrics JSON files if available.
    """
    try:
        with open("results/metrics/hybrid_model.json") as f:
            leaky = json.load(f)
    except FileNotFoundError:
        print("results/metrics/hybrid_model.json not found -- skipping 3-way comparison plot. "
              "Run 07_hybrid_model.py first if you want this figure.")
        return

    with open(METRICS_PATH) as f:
        coldstart = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    labels = ["SVD-Only", "Same-Review\nHybrid (leaky)", "Leave-One-Out\nHybrid (cold-start)"]
    colors = ["#a0a0a0", "#C44E52", "#55A868"]

    rmse_vals = [leaky["svd_baseline_rmse"], leaky["hybrid_rmse"], coldstart["hybrid_coldstart_rmse"]]
    mae_vals = [leaky["svd_baseline_mae"], leaky["hybrid_mae"], coldstart["hybrid_coldstart_mae"]]

    for ax, vals, title in zip(axes, [rmse_vals, mae_vals], ["RMSE", "MAE"]):
        ax.bar(labels, vals, color=colors)
        ax.set_ylabel(title)
        ax.set_title(f"{title} Comparison")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=9)

    fig.suptitle("Effect of Review-Text Leakage on Hybrid Model Performance")
    fig.tight_layout()
    path = os.path.join(save_dir, "leaky_vs_coldstart_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_metrics_by_rating(by_rating_df, save_dir):
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(by_rating_df))
    width = 0.35

    ax.bar(x - width / 2, by_rating_df["svd_only_rmse"], width,
           label="SVD-Only Baseline", color="#a0a0a0")
    ax.bar(x + width / 2, by_rating_df["coldstart_hybrid_rmse"], width,
           label="Cold-Start Hybrid", color="#55A868")

    ax.set_xlabel("Actual Star Rating")
    ax.set_ylabel("RMSE")
    ax.set_title("Cold-Start Hybrid: RMSE by Actual Rating Class")
    ax.set_xticks(x)
    ax.set_xticklabels(by_rating_df["rating"].astype(int))
    ax.legend()

    fig.tight_layout()
    path = os.path.join(save_dir, "coldstart_rmse_by_rating.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_precision_at_k(metrics, save_dir):
    small = metrics["precision_small_k"]
    full = metrics["precision_full_k"]

    fig, ax = plt.subplots(figsize=(7, 5))

    full_coverage_pct = full["coverage"]["coverage_pct_of_all_users"]
    labels = [f"Precision@{small['k']}\n(all users)",
              f"Precision@{full['k']}\n({full_coverage_pct}% of users)"]

    def _val(v):
        return 0.0 if v is None else v

    svd_vals = [_val(small["svd_precision"]), _val(full["svd_precision"])]
    hybrid_vals = [_val(small["hybrid_precision"]), _val(full["hybrid_precision"])]

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, svd_vals, width, label="SVD-Only Baseline", color="#a0a0a0")
    ax.bar(x + width / 2, hybrid_vals, width, label="Cold-Start Hybrid", color="#55A868")

    ax.set_ylabel("Precision")
    ax.set_title(f"Cold-Start Hybrid Precision@K\n(relevance threshold: rating >= {metrics['relevance_threshold']})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.legend()

    for i, (svd_v, hyb_v, svd_raw, hyb_raw) in enumerate(
            zip(svd_vals, hybrid_vals, [small["svd_precision"], full["svd_precision"]],
                [small["hybrid_precision"], full["hybrid_precision"]])):
        svd_label = "N/A" if svd_raw is None else f"{svd_v:.3f}"
        hyb_label = "N/A" if hyb_raw is None else f"{hyb_v:.3f}"
        ax.text(i - width / 2, svd_v + 0.02, svd_label, ha="center", fontsize=9)
        ax.text(i + width / 2, hyb_v + 0.02, hyb_label, ha="center", fontsize=9)

    fig.tight_layout()
    path = os.path.join(save_dir, "coldstart_precision_at_k_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def save_by_rating_table(by_rating_df):
    path = "results/tables/coldstart_evaluation_by_rating.csv"
    by_rating_df.to_csv(path, index=False)
    print(f"Saved: {path}")


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    df = load_predictions()
    metrics = load_metrics()
    importance_df = load_feature_importances()

    by_rating_df = compute_metrics_by_rating(df)
    save_by_rating_table(by_rating_df)

    plot_overall_comparison(metrics, FIGURES_DIR)
    plot_leaky_vs_coldstart_comparison(FIGURES_DIR)
    plot_metrics_by_rating(by_rating_df, FIGURES_DIR)
    plot_precision_at_k(metrics, FIGURES_DIR)

    print("\n" + "*" * 50)
    print("COLD-START EVALUATION COMPLETE")
    print(f"All figures saved to {FIGURES_DIR}/")
    print("*" * 50)