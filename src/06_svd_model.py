"""
File: 06_svd_model.py
Project: Hybrid Recommendation System

Purpose:
Trains the SVD collaborative-filtering baseline using the canonical
train/validation/test split produced by 02_sampling_and_split.py.

Hyperparameters (n_factors, n_epochs, lr_all, reg_all) are selected by
a small grid search evaluated on the VALIDATION split only (never the
test split), then the winning configuration is refit on the training
split and evaluated once on the untouched test split. This keeps the
reported test metrics free of any influence from the tuning process.
"""

import pandas as pd
import json
import pickle
import os

from surprise import Dataset, Reader, SVD
from surprise import accuracy

from configure import SENTIMENT_PATH, SVD_MODEL_PATH, RANDOM_STATE, SVD_PARAM_GRID


def load_data():
    df = pd.read_csv(SENTIMENT_PATH)
    print(f"Loaded dataset with shape: {df.shape}")
    return df


def build_sets(df):
    """
    Builds a Surprise trainset from the "train" rows, and plain testset-
    style (user, item, rating) tuple lists for the "val" and "test" rows,
    using the canonical split column rather than re-splitting independently.
    """
    print("\n" + "*" * 50)
    print("BUILDING TRAIN/VALIDATION/TEST SETS FROM CANONICAL SPLIT")
    print("*" * 50)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    print(f"Train rows: {len(train_df)}")
    print(f"Val rows:   {len(val_df)}")
    print(f"Test rows:  {len(test_df)}")

    reader = Reader(rating_scale=(1, 5))
    trainset = Dataset.load_from_df(
        train_df[["UserId", "ProductId", "Score"]], reader
    ).build_full_trainset()

    valset = list(val_df[["UserId", "ProductId", "Score"]].itertuples(index=False, name=None))
    testset = list(test_df[["UserId", "ProductId", "Score"]].itertuples(index=False, name=None))

    return trainset, valset, testset


def tune_hyperparameters(trainset, valset, param_grid=SVD_PARAM_GRID):
    """
    Small grid search: fits an SVD model on the training set for each
    candidate hyperparameter combination, scores it on the VALIDATION
    set, and returns the combination with the lowest validation RMSE.
    The test set is never touched during this process.
    """
    print("\n" + "*" * 50)
    print(f"TUNING SVD HYPERPARAMETERS ({len(param_grid)} candidates, scored on validation set)")
    print("*" * 50)

    best_params = None
    best_rmse = float("inf")
    results = []

    for params in param_grid:
        model = SVD(random_state=RANDOM_STATE, **params)
        model.fit(trainset)
        val_predictions = model.test(valset)
        val_rmse = accuracy.rmse(val_predictions, verbose=False)
        results.append({**params, "val_rmse": val_rmse})
        print(f"  {params} -> val RMSE: {val_rmse:.4f}")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_params = params

    print(f"\nBest hyperparameters (by validation RMSE): {best_params}")
    print(f"Best validation RMSE: {best_rmse:.4f}")

    return best_params, results


def train_final_model(trainset, params):
    print("\n" + "*" * 50)
    print(f"TRAINING FINAL SVD MODEL WITH TUNED HYPERPARAMETERS: {params}")
    print("*" * 50)

    model = SVD(random_state=RANDOM_STATE, **params)
    model.fit(trainset)

    print("Training complete.")
    return model


def evaluate_model(model, testset):
    print("\n" + "*" * 50)
    print("EVALUATING FINAL SVD MODEL (on canonical held-out TEST rows, never used for tuning)")
    print("*" * 50)

    predictions = model.test(testset)

    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)

    print(f"\nFinal Test RMSE: {rmse:.4f}")
    print(f"Final Test MAE: {mae:.4f}")

    return predictions, rmse, mae


if __name__ == "__main__":
    df = load_data()
    trainset, valset, testset = build_sets(df)

    best_params, tuning_results = tune_hyperparameters(trainset, valset)
    model = train_final_model(trainset, best_params)
    predictions, rmse, mae = evaluate_model(model, testset)

    predictions_df = pd.DataFrame(
        predictions, columns=["UserId", "ProductId", "Actual", "Predicted", "Details"]
    )
    os.makedirs("results/tables", exist_ok=True)
    predictions_df.to_csv("results/tables/svd_predictions.csv", index=False)
    print("Saved predictions to results/tables/svd_predictions.csv")

    pd.DataFrame(tuning_results).to_csv("results/tables/svd_hyperparameter_tuning.csv", index=False)
    print("Saved hyperparameter tuning results to results/tables/svd_hyperparameter_tuning.csv")

    os.makedirs("results/models", exist_ok=True)
    with open(SVD_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved trained model to {SVD_MODEL_PATH}")

    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/svd_baseline.json", "w") as f:
        json.dump({
            "rmse": rmse, "mae": mae,
            "n_train": trainset.n_ratings, "n_val": len(valset), "n_test": len(testset),
            "best_hyperparameters": best_params,
        }, f, indent=2)
    print("Saved baseline metrics to results/metrics/svd_baseline.json")