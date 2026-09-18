"""
File: 07b_hybrid_model_coldstart.py
Project: Hybrid Recommendation System

Purpose:
Builds a leakage-free variant of the hybrid model. Everywhere else in
this pipeline (07_hybrid_model.py), the TF-IDF content vector and VADER
sentiment score used to predict a review's Score are computed FROM THAT
SAME REVIEW's own text. That is a real methodological limitation,
disclosed in the dissertation's Discussion section: a genuine
recommender must predict a rating before the user has written the
review, so the review's own text is not actually available as a
predictive input in a real deployment.

This script fixes that. For every review being predicted, the content
and sentiment features are instead built from OTHER reviews of the same
product (a "leave-one-out" product profile) -- text that plausibly
already exists at prediction time, since other users will typically
have already reviewed a product before any given user does. This
simulates a genuine cold-start scenario: "given what people have said
about this product so far, and this user's rating history via SVD,
predict how this new review will score" -- without reading the answer
first.

Design notes:
  - The SVD collaborative-filtering feature is unchanged: it was
    already leakage-free, since it only uses the rating matrix, never
    review text.
  - The leave-one-out product aggregates are computed using ALL other
    reviews of a product regardless of train/val/test split, since
    those other reviews are independent, already-existing text in the
    real world -- this is NOT the same as leaking the target row's own
    label, and does not require the target row's own text or rating.
  - The TF-IDF vectorizer (vocabulary and IDF weights) is still the one
    fit on training rows only in 05_tfidf.py, to keep vocabulary
    leakage-free as before.
  - A fresh TruncatedSVD reducer is fit here, on the *aggregated*
    training-row vectors (not the original per-review vectors), since
    the feature being reduced is now the aggregate, not the raw
    per-review TF-IDF vector.
  - Products with no OTHER reviews (i.e. a single-review product) fall
    back to the training-set global mean sentiment / mean TF-IDF
    vector, computed from train rows only to avoid any val/test
    leakage into the fallback statistic itself.
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import scipy.sparse as sp

from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from configure import (
    SENTIMENT_PATH, TFIDF_MATRIX_PATH, SVD_MODEL_PATH, HYBRID_COLDSTART_MODEL_PATH,
    RANDOM_STATE, N_TFIDF_SVD_COMPONENTS, PRECISION_AT_K, PRECISION_AT_K_SMALL,
    RELEVANCE_THRESHOLD, RF_PARAM_GRID,
)


def load_data():
    df = pd.read_csv(SENTIMENT_PATH)
    print(f"Loaded dataset with shape: {df.shape}")
    return df


def load_tfidf_matrix():
    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)
    print(f"Loaded TF-IDF matrix with shape: {tfidf_matrix.shape}")
    return tfidf_matrix


def load_svd_model():
    with open(SVD_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Loaded trained (tuned) SVD baseline model.")
    return model


def get_svd_predictions(svd_model, df):
    """
    Unchanged from 07_hybrid_model.py: this feature was already
    leakage-free, since it derives from the rating matrix only, never
    from review text.
    """
    print("\n" + "*" * 50)
    print("GENERATING SVD PREDICTIONS AS A FEATURE")
    print("*" * 50)

    preds = [
        svd_model.predict(uid, iid).est
        for uid, iid in zip(df["UserId"], df["ProductId"])
    ]
    print("Sample SVD predictions:", preds[:5])
    return np.array(preds)


def build_leave_one_out_product_aggregates(df, tfidf_matrix, train_mask):
    """
    For every row, computes:
      - agg_vader: the mean VADER compound score across all OTHER
        reviews of the same product (excluding the row itself).
      - agg_tfidf: the mean TF-IDF vector across all OTHER reviews of
        the same product (excluding the row itself).

    Products with no other reviews fall back to the training-set global
    mean (computed from train rows only).

    Uses grouped sums so this is O(n) rather than O(n^2): each row's
    leave-one-out mean is (product_sum - own_value) / (product_count - 1).
    """
    print("\n" + "*" * 50)
    print("BUILDING LEAVE-ONE-OUT PRODUCT AGGREGATES (leakage-free content features)")
    print("*" * 50)

    n = len(df)
    products = df["ProductId"].values
    vader = df["vader_compound"].values

    # --- VADER aggregate ---
    product_vader_sum = df.groupby("ProductId")["vader_compound"].transform("sum").values
    product_count = df.groupby("ProductId")["vader_compound"].transform("count").values

    train_global_mean_vader = df.loc[train_mask, "vader_compound"].mean()

    denom = (product_count - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        agg_vader = (product_vader_sum - vader) / denom
    agg_vader = np.where(denom > 0, agg_vader, train_global_mean_vader)

    # --- TF-IDF aggregate (sparse-friendly) ---
    # Sum TF-IDF vectors per product using a product-indicator matmul,
    # then subtract each row's own vector and divide by (count - 1).
    product_codes, product_uniques = pd.factorize(df["ProductId"])
    n_products = len(product_uniques)

    indicator = sp.csr_matrix(
        (np.ones(n), (product_codes, np.arange(n))),
        shape=(n_products, n),
    )
    product_tfidf_sum = indicator.dot(tfidf_matrix)  # (n_products, n_features)
    row_product_sum = product_tfidf_sum[product_codes, :]  # broadcast back to (n, n_features)

    train_global_mean_tfidf = tfidf_matrix[train_mask].mean(axis=0)
    train_global_mean_tfidf = np.asarray(train_global_mean_tfidf)  # dense (1, n_features)

    agg_tfidf = sp.lil_matrix(tfidf_matrix.shape, dtype=np.float64)
    has_others = denom > 0
    n_features = tfidf_matrix.shape[1]

    # Rows with at least one other review of the same product: leave-one-out mean.
    idx_has_others = np.where(has_others)[0]
    if len(idx_has_others) > 0:
        numer = (row_product_sum[idx_has_others] - tfidf_matrix[idx_has_others]).tocsr()
        d = denom[idx_has_others].reshape(-1, 1)
        agg_tfidf[idx_has_others] = numer.multiply(1.0 / d)

    # Rows with no other review of the same product: fall back to the
    # training-set global mean TF-IDF vector.
    idx_no_others = np.where(~has_others)[0]
    if len(idx_no_others) > 0:
        fallback_block = np.repeat(train_global_mean_tfidf, len(idx_no_others), axis=0)
        agg_tfidf[idx_no_others] = fallback_block

    agg_tfidf = agg_tfidf.tocsr()

    print(f"Rows with >=1 other review of the same product: {has_others.sum()} / {n}")
    print(f"Rows falling back to train-set global mean (single-review products): {(~has_others).sum()} / {n}")

    return agg_vader, agg_tfidf


def reduce_aggregated_tfidf(agg_tfidf, train_mask, n_components=N_TFIDF_SVD_COMPONENTS):
    """
    Fits a fresh TruncatedSVD reducer on the AGGREGATED training-row
    vectors (not the raw per-review vectors used in 07_hybrid_model.py),
    since the feature being reduced here is the leave-one-out product
    aggregate, not the review's own TF-IDF vector.
    """
    print("\n" + "*" * 50)
    print("REDUCING AGGREGATED TF-IDF DIMENSIONALITY (fit on train rows only)")
    print("*" * 50)

    reducer = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    reducer.fit(agg_tfidf[train_mask, :])
    reduced = reducer.transform(agg_tfidf)

    explained_var = reducer.explained_variance_ratio_.sum()
    print(f"Reduced aggregated TF-IDF shape: {reduced.shape}")
    print(f"Explained variance retained (on train rows): {explained_var:.4f}")

    return reduced


def build_feature_matrix(df, svd_preds, agg_vader, agg_tfidf_reduced):
    print("\n" + "*" * 50)
    print("BUILDING LEAKAGE-FREE HYBRID FEATURE MATRIX")
    print("*" * 50)

    svd_preds = svd_preds.reshape(-1, 1)
    agg_vader = agg_vader.reshape(-1, 1)

    X = np.hstack([svd_preds, agg_vader, agg_tfidf_reduced])
    y = df["Score"].values

    feature_names = ["svd_pred", "agg_vader_compound"] + [
        f"agg_tfidf_svd_{i}" for i in range(agg_tfidf_reduced.shape[1])
    ]

    print(f"Final feature matrix shape: {X.shape}")
    return X, y, feature_names


def tune_hyperparameters(X_train, y_train, X_val, y_val, param_grid=RF_PARAM_GRID):
    print("\n" + "*" * 50)
    print(f"TUNING RANDOM FOREST HYPERPARAMETERS ({len(param_grid)} candidates, scored on validation set)")
    print("*" * 50)

    best_params = None
    best_rmse = float("inf")
    results = []

    for params in param_grid:
        model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        results.append({**params, "val_rmse": val_rmse})
        print(f"  {params} -> val RMSE: {val_rmse:.4f}")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_params = params

    print(f"\nBest hyperparameters (by validation RMSE): {best_params}")
    print(f"Best validation RMSE: {best_rmse:.4f}")

    return best_params, results


def train_final_model(X_train, y_train, params):
    print("\n" + "*" * 50)
    print(f"TRAINING FINAL LEAKAGE-FREE HYBRID MODEL: {params}")
    print("*" * 50)

    model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    model.fit(X_train, y_train)

    print("Training complete.")
    return model


def evaluate_model(model, X_test, y_test):
    print("\n" + "*" * 50)
    print("EVALUATING LEAKAGE-FREE HYBRID MODEL (on canonical held-out test rows)")
    print("*" * 50)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Leakage-Free Hybrid RMSE: {rmse:.4f}")
    print(f"Leakage-Free Hybrid MAE:  {mae:.4f}")

    return y_pred, rmse, mae


def evaluate_svd_baseline(X_test, y_test):
    print("\n" + "*" * 50)
    print("EVALUATING SVD-ONLY BASELINE (FOR COMPARISON, SAME AS 07_hybrid_model.py)")
    print("*" * 50)

    svd_only_preds = X_test[:, 0]
    rmse = np.sqrt(mean_squared_error(y_test, svd_only_preds))
    mae = mean_absolute_error(y_test, svd_only_preds)

    print(f"SVD-Only Baseline RMSE: {rmse:.4f}")
    print(f"SVD-Only Baseline MAE:  {mae:.4f}")

    return svd_only_preds, rmse, mae


def get_feature_importances(model, feature_names):
    print("\n" + "*" * 50)
    print("FEATURE IMPORTANCE SUMMARY")
    print("*" * 50)

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names, "importance": importances
    }).sort_values("importance", ascending=False)

    svd_imp = importance_df.loc[importance_df["feature"] == "svd_pred", "importance"].sum()
    vader_imp = importance_df.loc[importance_df["feature"] == "agg_vader_compound", "importance"].sum()
    tfidf_imp = importance_df.loc[
        importance_df["feature"].str.startswith("agg_tfidf_svd_"), "importance"
    ].sum()

    print(f"SVD prediction (collaborative signal): {svd_imp:.4f}")
    print(f"Aggregated VADER sentiment (other reviews): {vader_imp:.4f}")
    print(f"Aggregated TF-IDF components (other reviews, aggregated): {tfidf_imp:.4f}")
    print("\nTop 10 individual features:")
    print(importance_df.head(10).to_string(index=False))

    return importance_df


def precision_at_k(user_ids, y_true, y_pred, k, relevance_threshold=RELEVANCE_THRESHOLD,
                    require_full_k=False):
    """
    Identical logic to 07_hybrid_model.py's precision_at_k -- see that
    file for full documentation of the two-mode design.
    """
    eval_df = pd.DataFrame({"UserId": user_ids, "y_true": y_true, "y_pred": y_pred})

    n_total_users = eval_df["UserId"].nunique()
    n_with_relevant = 0
    rows = []

    for user_id, group in eval_df.groupby("UserId"):
        n_relevant = (group["y_true"] >= relevance_threshold).sum()
        if n_relevant == 0:
            continue
        n_with_relevant += 1

        if require_full_k and len(group) < k:
            continue

        top_k = group.sort_values("y_pred", ascending=False).head(k)
        n_relevant_in_top_k = (top_k["y_true"] >= relevance_threshold).sum()
        precision = n_relevant_in_top_k / len(top_k)

        rows.append({
            "UserId": user_id, "n_test_items": len(group), "n_relevant": n_relevant,
            "k_used": len(top_k), "precision_at_k": precision,
        })

    per_user_df = pd.DataFrame(rows)
    n_included = len(per_user_df)
    mean_precision = per_user_df["precision_at_k"].mean() if n_included else float("nan")

    coverage = {
        "k": k, "require_full_k": require_full_k,
        "n_total_users": int(n_total_users),
        "n_users_with_relevant_item": int(n_with_relevant),
        "n_users_included": int(n_included),
        "coverage_pct_of_all_users": round(100 * n_included / n_total_users, 2) if n_total_users else 0.0,
    }

    return mean_precision, per_user_df, coverage


def save_outputs(model, rmse, mae, y_test, y_pred, svd_baseline_preds,
                  svd_baseline_rmse, svd_baseline_mae, importance_df, best_params,
                  hybrid_small_k, svd_small_k, small_k_coverage,
                  hybrid_full_k, svd_full_k, full_k_coverage):
    os.makedirs("results/models", exist_ok=True)
    with open(HYBRID_COLDSTART_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved leakage-free hybrid model to {HYBRID_COLDSTART_MODEL_PATH}")

    rmse_improvement_pct = 100 * (svd_baseline_rmse - rmse) / svd_baseline_rmse
    mae_improvement_pct = 100 * (svd_baseline_mae - mae) / svd_baseline_mae

    def _clean(v):
        try:
            return None if (isinstance(v, float) and np.isnan(v)) else v
        except TypeError:
            return v

    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/hybrid_model_coldstart.json", "w") as f:
        json.dump({
            "hybrid_coldstart_rmse": rmse,
            "hybrid_coldstart_mae": mae,
            "svd_baseline_rmse": svd_baseline_rmse,
            "svd_baseline_mae": svd_baseline_mae,
            "rmse_improvement_pct": rmse_improvement_pct,
            "mae_improvement_pct": mae_improvement_pct,
            "best_hyperparameters": best_params,
            "relevance_threshold": RELEVANCE_THRESHOLD,
            "precision_small_k": {
                "k": PRECISION_AT_K_SMALL,
                "hybrid_precision": _clean(hybrid_small_k),
                "svd_precision": _clean(svd_small_k),
                "coverage": small_k_coverage,
            },
            "precision_full_k": {
                "k": PRECISION_AT_K,
                "hybrid_precision": _clean(hybrid_full_k),
                "svd_precision": _clean(svd_full_k),
                "coverage": full_k_coverage,
            },
        }, f, indent=2)
    print("Saved leakage-free hybrid metrics to results/metrics/hybrid_model_coldstart.json")
    print(f"  -> RMSE improvement over SVD-only: {rmse_improvement_pct:.2f}%")
    print(f"  -> MAE improvement over SVD-only:  {mae_improvement_pct:.2f}%")

    os.makedirs("results/tables", exist_ok=True)
    pd.DataFrame({
        "Actual": y_test, "Predicted": y_pred, "SVD_Only_Predicted": svd_baseline_preds,
    }).to_csv("results/tables/hybrid_coldstart_predictions.csv", index=False)
    print("Saved leakage-free hybrid predictions to results/tables/hybrid_coldstart_predictions.csv")

    importance_df.to_csv("results/tables/hybrid_coldstart_feature_importances.csv", index=False)
    print("Saved feature importances to results/tables/hybrid_coldstart_feature_importances.csv")


if __name__ == "__main__":
    df = load_data()
    tfidf_matrix = load_tfidf_matrix()
    svd_model = load_svd_model()

    train_mask = (df["split"] == "train").values
    val_mask = (df["split"] == "val").values
    test_mask = (df["split"] == "test").values

    svd_preds = get_svd_predictions(svd_model, df)
    agg_vader, agg_tfidf = build_leave_one_out_product_aggregates(df, tfidf_matrix, train_mask)
    agg_tfidf_reduced = reduce_aggregated_tfidf(agg_tfidf, train_mask)

    X, y, feature_names = build_feature_matrix(df, svd_preds, agg_vader, agg_tfidf_reduced)

    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]
    user_ids_test = df.loc[test_mask, "UserId"].values

    best_params, tuning_results = tune_hyperparameters(X_train, y_train, X_val, y_val)
    model = train_final_model(X_train, y_train, best_params)
    y_pred, rmse, mae = evaluate_model(model, X_test, y_test)
    svd_baseline_preds, svd_baseline_rmse, svd_baseline_mae = evaluate_svd_baseline(X_test, y_test)
    importance_df = get_feature_importances(model, feature_names)

    hybrid_small_k, hybrid_small_df, small_k_coverage = precision_at_k(
        user_ids_test, y_test, y_pred, k=PRECISION_AT_K_SMALL, require_full_k=False)
    svd_small_k, svd_small_df, _ = precision_at_k(
        user_ids_test, y_test, svd_baseline_preds, k=PRECISION_AT_K_SMALL, require_full_k=False)

    hybrid_full_k, hybrid_full_df, full_k_coverage = precision_at_k(
        user_ids_test, y_test, y_pred, k=PRECISION_AT_K, require_full_k=True)
    svd_full_k, svd_full_df, _ = precision_at_k(
        user_ids_test, y_test, svd_baseline_preds, k=PRECISION_AT_K, require_full_k=True)

    os.makedirs("results/tables", exist_ok=True)
    pd.DataFrame(tuning_results).to_csv("results/tables/hybrid_coldstart_hyperparameter_tuning.csv", index=False)

    save_outputs(
        model, rmse, mae, y_test, y_pred, svd_baseline_preds,
        svd_baseline_rmse, svd_baseline_mae, importance_df, best_params,
        hybrid_small_k, svd_small_k, small_k_coverage,
        hybrid_full_k, svd_full_k, full_k_coverage,
    )