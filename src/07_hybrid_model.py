"""
File: 07_hybrid_model.py
Project: Hybrid Recommendation System

Purpose:
Builds the hybrid feature matrix (SVD prediction + VADER sentiment +
reduced TF-IDF) and trains a Random Forest meta-learner on top of it.

Changes from the original version of this script:
  1. Uses the CANONICAL train/validation/test split (the "split" column
     produced by 02_sampling_and_split.py) instead of drawing its own
     independent sample and re-splitting with sklearn.train_test_split.
     This is what makes the SVD-only baseline comparison in this script
     a genuinely fair, held-out comparison against 06_svd_model.py.
  2. The TruncatedSVD dimensionality reducer for the TF-IDF matrix is
     FIT ONLY on training rows and used to transform validation/test
     rows, avoiding a subtle leakage channel where held-out row content
     could influence the reduced feature space.
  3. Random Forest hyperparameters (n_estimators, max_depth) are chosen
     by a small grid search scored on the VALIDATION split, then the
     winning configuration is refit on the training split and evaluated
     once on the untouched test split.
  4. Adds Precision@K, satisfying Proposal Objective 6, which named both
     RMSE/MAE and Precision@K as evaluation metrics.
"""

import pandas as pd
import numpy as np
import os
import json
import pickle

from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from configure import (
    SENTIMENT_PATH, TFIDF_MATRIX_PATH, SVD_MODEL_PATH, HYBRID_MODEL_PATH,
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
    print("Loaded trained SVD baseline model.")
    return model


def get_svd_predictions(svd_model, df):
    """
    Uses the trained SVD baseline model (trained only on the canonical
    training rows, in 06_svd_model.py) to predict a rating for every
    (UserId, ProductId) pair in the dataset. Unknown users/items fall
    back to Surprise's default (global mean) estimate automatically.
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


def reduce_tfidf(tfidf_matrix, train_mask, n_components=N_TFIDF_SVD_COMPONENTS):
    """
    Reduces the TF-IDF matrix to n_components via TruncatedSVD, fitting
    ONLY on the training rows and transforming the full matrix (train +
    test) with that fitted reducer. This avoids letting the structure of
    test-set reviews influence the reduced feature space.
    """
    print("\n" + "*" * 50)
    print("REDUCING TF-IDF DIMENSIONALITY (fit on train rows only)")
    print("*" * 50)

    reducer = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    reducer.fit(tfidf_matrix[train_mask, :])

    tfidf_reduced = reducer.transform(tfidf_matrix)

    explained_var = reducer.explained_variance_ratio_.sum()
    print(f"Reduced TF-IDF shape: {tfidf_reduced.shape}")
    print(f"Explained variance retained (on train rows): {explained_var:.4f}")

    return tfidf_reduced


def build_feature_matrix(df, svd_preds, tfidf_reduced):
    """
    Combines SVD predictions, VADER compound scores, and reduced TF-IDF
    features into a single feature matrix, aligned row-for-row with df.
    """
    print("\n" + "*" * 50)
    print("BUILDING HYBRID FEATURE MATRIX")
    print("*" * 50)

    vader_scores = df["vader_compound"].values.reshape(-1, 1)
    svd_preds = svd_preds.reshape(-1, 1)

    X = np.hstack([svd_preds, vader_scores, tfidf_reduced])
    y = df["Score"].values

    feature_names = ["svd_pred", "vader_compound"] + [
        f"tfidf_svd_{i}" for i in range(tfidf_reduced.shape[1])
    ]

    print(f"Final feature matrix shape: {X.shape}")
    return X, y, feature_names


def tune_hyperparameters(X_train, y_train, X_val, y_val, param_grid=RF_PARAM_GRID):
    """
    Small grid search over Random Forest hyperparameters, scored on the
    VALIDATION split. The test split is never touched during this
    process, so the final reported test metrics are not influenced by
    the tuning search.
    """
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


def train_hybrid_model(X_train, y_train, params):
    print("\n" + "*" * 50)
    print(f"TRAINING FINAL HYBRID MODEL WITH TUNED HYPERPARAMETERS: {params}")
    print("*" * 50)

    model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    model.fit(X_train, y_train)

    print("Training complete.")
    return model


def evaluate_hybrid_model(model, X_test, y_test):
    print("\n" + "*" * 50)
    print("EVALUATING HYBRID MODEL (on canonical held-out test rows)")
    print("*" * 50)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Hybrid Model RMSE: {rmse:.4f}")
    print(f"Hybrid Model MAE:  {mae:.4f}")

    return y_pred, rmse, mae


def evaluate_svd_baseline(X_test, y_test):
    """
    Evaluates the raw SVD prediction alone (column 0 of X_test) on the
    same canonical held-out rows used for the hybrid model, so the two
    are a genuinely fair comparison.
    """
    print("\n" + "*" * 50)
    print("EVALUATING SVD-ONLY BASELINE (FOR COMPARISON)")
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
    vader_imp = importance_df.loc[importance_df["feature"] == "vader_compound", "importance"].sum()
    tfidf_imp = importance_df.loc[
        importance_df["feature"].str.startswith("tfidf_svd_"), "importance"
    ].sum()

    print(f"SVD prediction (collaborative signal): {svd_imp:.4f}")
    print(f"VADER sentiment (compound score):      {vader_imp:.4f}")
    print(f"TF-IDF components (content signal, aggregated): {tfidf_imp:.4f}")
    print("\nTop 10 individual features:")
    print(importance_df.head(10).to_string(index=False))

    return importance_df


def precision_at_k(user_ids, y_true, y_pred, k=PRECISION_AT_K,
                    relevance_threshold=RELEVANCE_THRESHOLD, require_full_k=False):
    """
    Computes Precision@K, following the standard rating-threshold
    approach: for each user in the test set, items with true rating >=
    relevance_threshold are considered "relevant". Test items are
    ranked by PREDICTED rating (descending); Precision@K for that user
    is the fraction of their top-K predicted items that are actually
    relevant.

    Sparse review datasets typically give most users very few test-set
    items -- often fewer than K. When that happens, "top-K" silently
    becomes "all of the user's available items", and Precision@K stops
    testing ranking ability at all (every model gets the same score,
    since there is no ranking decision left to make). Two modes are
    provided to handle this honestly rather than silently:

      require_full_k=False (default): every user with at least one
      relevant item is scored on however many test items they have
      (a common, documented adjustment -- see the Surprise library
      FAQ's precision/recall-at-k recipe). This maximises coverage but
      can understate the metric's meaning when K exceeds most users'
      item counts -- intended to be used with a small K.

      require_full_k=True: only users with at least K test items are
      scored, so every included user's Precision@K reflects an actual
      top-K ranking decision. This is a stricter, more meaningful test
      of ranking quality, at the cost of excluding users who don't have
      enough test data -- the exclusion rate is reported so this
      trade-off is visible rather than hidden.

    Returns the mean Precision@K, the per-user detail table, and a
    coverage dict describing how many users were included/excluded.
    """
    mode_label = "full-K subset (users with >= K test items)" if require_full_k else "all users with >=1 relevant item"
    print("\n" + "*" * 50)
    print(f"EVALUATING PRECISION@{k} (relevance threshold: rating >= {relevance_threshold}, mode: {mode_label})")
    print("*" * 50)

    eval_df = pd.DataFrame({
        "UserId": user_ids,
        "y_true": y_true,
        "y_pred": y_pred,
    })

    n_total_users = eval_df["UserId"].nunique()
    n_with_relevant = 0
    rows = []

    for user_id, group in eval_df.groupby("UserId"):
        n_relevant = (group["y_true"] >= relevance_threshold).sum()
        if n_relevant == 0:
            continue  # Precision@K is undefined with no relevant items
        n_with_relevant += 1

        if require_full_k and len(group) < k:
            continue  # not enough test items for a genuine top-K decision

        top_k = group.sort_values("y_pred", ascending=False).head(k)
        n_relevant_in_top_k = (top_k["y_true"] >= relevance_threshold).sum()
        precision = n_relevant_in_top_k / len(top_k)

        rows.append({
            "UserId": user_id,
            "n_test_items": len(group),
            "n_relevant": n_relevant,
            "k_used": len(top_k),
            "precision_at_k": precision,
        })

    per_user_df = pd.DataFrame(rows)
    n_included = len(per_user_df)
    mean_precision = per_user_df["precision_at_k"].mean() if n_included else float("nan")

    coverage = {
        "k": k,
        "require_full_k": require_full_k,
        "n_total_users": int(n_total_users),
        "n_users_with_relevant_item": int(n_with_relevant),
        "n_users_included": int(n_included),
        "coverage_pct_of_all_users": round(100 * n_included / n_total_users, 2) if n_total_users else 0.0,
    }

    print(f"Users with >=1 relevant test item: {n_with_relevant} / {n_total_users}")
    print(f"Users included in this Precision@{k} calculation: {n_included} "
          f"({coverage['coverage_pct_of_all_users']}% of all test users)")
    print(f"Mean Precision@{k}: {mean_precision:.4f}")

    return mean_precision, per_user_df, coverage


def save_outputs(model, rmse, mae, y_test, y_pred, svd_baseline_preds,
                  svd_baseline_rmse, svd_baseline_mae, importance_df,
                  hybrid_small_k, svd_small_k, small_k_coverage,
                  hybrid_full_k, svd_full_k, full_k_coverage, best_params):
    os.makedirs("results/models", exist_ok=True)
    with open(HYBRID_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved hybrid model to {HYBRID_MODEL_PATH}")

    rmse_improvement_pct = 100 * (svd_baseline_rmse - rmse) / svd_baseline_rmse
    mae_improvement_pct = 100 * (svd_baseline_mae - mae) / svd_baseline_mae

    os.makedirs("results/metrics", exist_ok=True)

    def _clean(v):
        """json.dump writes NaN as a non-standard literal; store None instead
        so the metrics file stays valid JSON if a K value has zero coverage."""
        try:
            return None if (isinstance(v, float) and np.isnan(v)) else v
        except TypeError:
            return v

    with open("results/metrics/hybrid_model.json", "w") as f:
        json.dump({
            "hybrid_rmse": rmse,
            "hybrid_mae": mae,
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
    print("Saved hybrid metrics (with baseline + dual Precision@K comparison) to "
          "results/metrics/hybrid_model.json")
    print(f"  -> RMSE improvement over SVD-only: {rmse_improvement_pct:.2f}%")
    print(f"  -> MAE improvement over SVD-only:  {mae_improvement_pct:.2f}%")
    print(f"  -> Precision@{PRECISION_AT_K_SMALL} (all users): "
          f"Hybrid {hybrid_small_k:.4f} vs SVD {svd_small_k:.4f}")
    if full_k_coverage["n_users_included"] == 0:
        print(f"  -> Precision@{PRECISION_AT_K}: NO users had >= {PRECISION_AT_K} test items "
              f"(0.0% coverage) -- this K is too large for this dataset's sparsity; "
              f"consider lowering PRECISION_AT_K in configure.py and re-running.")
    else:
        print(f"  -> Precision@{PRECISION_AT_K} (subset with >= {PRECISION_AT_K} test items, "
              f"{full_k_coverage['coverage_pct_of_all_users']}% of users): "
              f"Hybrid {hybrid_full_k:.4f} vs SVD {svd_full_k:.4f}")

    os.makedirs("results/tables", exist_ok=True)
    pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred,
        "SVD_Only_Predicted": svd_baseline_preds,
    }).to_csv("results/tables/hybrid_predictions.csv", index=False)
    print("Saved hybrid predictions (with SVD-only baseline) to "
          "results/tables/hybrid_predictions.csv")

    importance_df.to_csv("results/tables/hybrid_feature_importances.csv", index=False)
    print("Saved feature importances to results/tables/hybrid_feature_importances.csv")


if __name__ == "__main__":
    df = load_data()
    tfidf_matrix = load_tfidf_matrix()
    svd_model = load_svd_model()

    train_mask = (df["split"] == "train").values
    val_mask = (df["split"] == "val").values
    test_mask = (df["split"] == "test").values

    svd_preds = get_svd_predictions(svd_model, df)
    tfidf_reduced = reduce_tfidf(tfidf_matrix, train_mask)

    X, y, feature_names = build_feature_matrix(df, svd_preds, tfidf_reduced)

    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]
    user_ids_test = df.loc[test_mask, "UserId"].values

    best_params, tuning_results = tune_hyperparameters(X_train, y_train, X_val, y_val)
    model = train_hybrid_model(X_train, y_train, best_params)
    y_pred, rmse, mae = evaluate_hybrid_model(model, X_test, y_test)
    svd_baseline_preds, svd_baseline_rmse, svd_baseline_mae = evaluate_svd_baseline(X_test, y_test)
    importance_df = get_feature_importances(model, feature_names)

    # Small K, all users with a relevant item -- maximum coverage.
    hybrid_small_k, hybrid_small_df, small_k_coverage = precision_at_k(
        user_ids_test, y_test, y_pred, k=PRECISION_AT_K_SMALL, require_full_k=False)
    svd_small_k, svd_small_df, _ = precision_at_k(
        user_ids_test, y_test, svd_baseline_preds, k=PRECISION_AT_K_SMALL, require_full_k=False)

    # Full K, only users with enough test items for a genuine top-K ranking test.
    hybrid_full_k, hybrid_full_df, full_k_coverage = precision_at_k(
        user_ids_test, y_test, y_pred, k=PRECISION_AT_K, require_full_k=True)
    svd_full_k, svd_full_df, _ = precision_at_k(
        user_ids_test, y_test, svd_baseline_preds, k=PRECISION_AT_K, require_full_k=True)

    os.makedirs("results/tables", exist_ok=True)
    pd.DataFrame(tuning_results).to_csv("results/tables/hybrid_hyperparameter_tuning.csv", index=False)
    print("Saved hyperparameter tuning results to results/tables/hybrid_hyperparameter_tuning.csv")

    hybrid_small_df.to_csv(f"results/tables/hybrid_precision_at_{PRECISION_AT_K_SMALL}_by_user.csv", index=False)
    svd_small_df.to_csv(f"results/tables/svd_precision_at_{PRECISION_AT_K_SMALL}_by_user.csv", index=False)
    hybrid_full_df.to_csv(f"results/tables/hybrid_precision_at_{PRECISION_AT_K}_by_user.csv", index=False)
    svd_full_df.to_csv(f"results/tables/svd_precision_at_{PRECISION_AT_K}_by_user.csv", index=False)

    save_outputs(
        model, rmse, mae, y_test, y_pred, svd_baseline_preds,
        svd_baseline_rmse, svd_baseline_mae, importance_df,
        hybrid_small_k, svd_small_k, small_k_coverage,
        hybrid_full_k, svd_full_k, full_k_coverage, best_params,
    )