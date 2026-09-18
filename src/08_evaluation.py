"""
File: 08_evaluation.py
Project: Hybrid Recommendation System

Purpose:
Loads the saved predictions, metrics, and feature importances from
07_hybrid_model.py and produces the diagnostic tables and figures used
in the dissertation's Results section, including the Precision@K
comparison (Proposal Objective 6).

A note on scope, since it matters for how these results should be read:
the "Predicted" (hybrid) and "SVD_Only_Predicted" (baseline) columns
come from 07_hybrid_model.py's canonical held-out test split. The
hybrid model's TF-IDF and VADER features are derived from the SAME
review whose Score is being predicted, so this evaluates "does the
review text match its own rating" more than it evaluates cold-start
recommendation quality. Keep that in mind when interpreting the
RMSE/MAE gap below.
"""

import pandas as pd
import numpy as np
import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configure import FIGURES_DIR

PREDICTIONS_PATH = "results/tables/hybrid_predictions.csv"
METRICS_PATH = "results/metrics/hybrid_model.json"
IMPORTANCE_PATH = "results/tables/hybrid_feature_importances.csv"


def load_predictions():
    df = pd.read_csv(PREDICTIONS_PATH)
    print(f"Loaded predictions with shape: {df.shape}")

    required_cols = {"Actual", "Predicted", "SVD_Only_Predicted"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns {missing} in {PREDICTIONS_PATH}. "
            "Re-run src/07_hybrid_model.py to regenerate it."
        )
    return df


def load_metrics():
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    print("Loaded headline metrics:", metrics)
    return metrics


def load_feature_importances():
    df = pd.read_csv(IMPORTANCE_PATH)
    print(f"Loaded feature importances with shape: {df.shape}")
    return df


def compute_metrics_by_rating(df):
    print("\n" + "*" * 50)
    print("METRICS BY RATING CLASS")
    print("*" * 50)

    rows = []
    for rating, group in df.groupby("Actual"):
        hybrid_rmse = np.sqrt(np.mean((group["Actual"] - group["Predicted"]) ** 2))
        hybrid_mae = np.mean(np.abs(group["Actual"] - group["Predicted"]))
        svd_rmse = np.sqrt(np.mean((group["Actual"] - group["SVD_Only_Predicted"]) ** 2))
        svd_mae = np.mean(np.abs(group["Actual"] - group["SVD_Only_Predicted"]))

        rows.append({
            "rating": rating,
            "n_samples": len(group),
            "hybrid_rmse": hybrid_rmse,
            "hybrid_mae": hybrid_mae,
            "svd_only_rmse": svd_rmse,
            "svd_only_mae": svd_mae,
        })

    by_rating_df = pd.DataFrame(rows)
    print(by_rating_df.to_string(index=False))
    return by_rating_df


def plot_overall_comparison(metrics, save_dir):
    fig, ax = plt.subplots(figsize=(6, 5))

    labels = ["RMSE", "MAE"]
    hybrid_vals = [metrics["hybrid_rmse"], metrics["hybrid_mae"]]
    svd_vals = [metrics["svd_baseline_rmse"], metrics["svd_baseline_mae"]]

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, svd_vals, width, label="SVD-Only Baseline", color="#a0a0a0")
    ax.bar(x + width / 2, hybrid_vals, width, label="Hybrid Model", color="#4C72B0")

    ax.set_ylabel("Error")
    ax.set_title("Hybrid Model vs. SVD-Only Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for i, (svd_v, hyb_v) in enumerate(zip(svd_vals, hybrid_vals)):
        ax.text(i - width / 2, svd_v + 0.01, f"{svd_v:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, hyb_v + 0.01, f"{hyb_v:.3f}", ha="center", fontsize=9)

    fig.tight_layout()
    path = os.path.join(save_dir, "overall_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_precision_at_k(metrics, save_dir):
    """
    Bar chart comparing Precision@K for the hybrid model vs. the
    SVD-only baseline, at two K values: a small K measured across all
    users with a relevant item, and a larger, conventional K measured
    only on the subset of users with enough test items for that K to
    reflect a genuine top-K ranking decision (coverage of that subset
    is annotated on the chart).
    """
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
    ax.bar(x + width / 2, hybrid_vals, width, label="Hybrid Model", color="#4C72B0")

    ax.set_ylabel("Precision")
    ax.set_title(f"Precision@K Comparison\n(relevance threshold: rating >= {metrics['relevance_threshold']})")
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
    path = os.path.join(save_dir, "precision_at_k_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_metrics_by_rating(by_rating_df, save_dir):
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(by_rating_df))
    width = 0.35

    ax.bar(x - width / 2, by_rating_df["svd_only_rmse"], width,
           label="SVD-Only Baseline", color="#a0a0a0")
    ax.bar(x + width / 2, by_rating_df["hybrid_rmse"], width,
           label="Hybrid Model", color="#4C72B0")

    ax.set_xlabel("Actual Star Rating")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE by Actual Rating Class")
    ax.set_xticks(x)
    ax.set_xticklabels(by_rating_df["rating"].astype(int))
    ax.legend()

    fig.tight_layout()
    path = os.path.join(save_dir, "rmse_by_rating.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_predicted_vs_actual(df, save_dir):
    fig, ax = plt.subplots(figsize=(6, 6))

    hb = ax.hexbin(df["Actual"], df["Predicted"], gridsize=30, cmap="Blues", mincnt=1)
    ax.plot([1, 5], [1, 5], color="red", linestyle="--", linewidth=1, label="Perfect prediction")

    ax.set_xlabel("Actual Rating")
    ax.set_ylabel("Predicted Rating")
    ax.set_title("Hybrid Model: Predicted vs. Actual")
    ax.legend(loc="upper left")
    fig.colorbar(hb, ax=ax, label="Count")

    fig.tight_layout()
    path = os.path.join(save_dir, "predicted_vs_actual.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_residual_distributions(df, save_dir):
    hybrid_resid = df["Predicted"] - df["Actual"]
    svd_resid = df["SVD_Only_Predicted"] - df["Actual"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(-4, 4, 60)

    ax.hist(svd_resid, bins=bins, alpha=0.5, label="SVD-Only Baseline", color="#a0a0a0")
    ax.hist(hybrid_resid, bins=bins, alpha=0.5, label="Hybrid Model", color="#4C72B0")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)

    ax.set_xlabel("Residual (Predicted - Actual)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")
    ax.legend()

    fig.tight_layout()
    path = os.path.join(save_dir, "residual_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_feature_importances(importance_df, save_dir, top_n=15):
    top_features = importance_df.head(top_n).sort_values("importance")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top_features["feature"], top_features["importance"], color="#4C72B0")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)")

    fig.tight_layout()
    path = os.path.join(save_dir, "feature_importances.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def save_by_rating_table(by_rating_df):
    os.makedirs("results/tables", exist_ok=True)
    path = "results/tables/evaluation_by_rating.csv"
    by_rating_df.to_csv(path, index=False)
    print(f"Saved: {path}")


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df = load_predictions()
    metrics = load_metrics()
    importance_df = load_feature_importances()

    by_rating_df = compute_metrics_by_rating(df)
    save_by_rating_table(by_rating_df)

    plot_overall_comparison(metrics, FIGURES_DIR)
    plot_precision_at_k(metrics, FIGURES_DIR)
    plot_metrics_by_rating(by_rating_df, FIGURES_DIR)
    plot_predicted_vs_actual(df, FIGURES_DIR)
    plot_residual_distributions(df, FIGURES_DIR)
    plot_feature_importances(importance_df, FIGURES_DIR)

    print("\n" + "*" * 50)
    print("EVALUATION COMPLETE")
    print(f"All figures saved to {FIGURES_DIR}/")
    print("*" * 50)