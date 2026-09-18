"""
File: 02_sampling_and_split.py
Project: Hybrid Recommendation System

Purpose:
Draws the single, canonical stratified sample and train/validation/test
split used by every model in this project (SVD baseline, TF-IDF/dimen-
sionality reduction, and the hybrid Random Forest).

This script exists to fix a data-leakage issue in the original pipeline:
previously, the SVD baseline script and the hybrid model script each
independently drew a "random_state=42" stratified sample and then split
it differently. Because both samples were drawn from the same underlying
file in the same order with the same seed, they substantially overlapped
-- meaning the hybrid model's "SVD-only baseline" comparison was likely
partly evaluated on rows the SVD model had already trained on, making
that baseline look artificially strong.

By producing ONE split here and having every downstream script load it,
the SVD baseline and the hybrid model are guaranteed to be trained on
the same training rows and evaluated on the same, genuinely unseen test
rows.

The split is now three-way (train / validation / test, 70/15/15):
the validation set is used exclusively for hyperparameter selection
(see 06_svd_model.py and 07_hybrid_model.py), keeping the test set
completely untouched until final evaluation.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from configure import (
    CLEAN_REVIEWS_PATH, SPLIT_REVIEWS_PATH,
    SAMPLE_SIZE, RANDOM_STATE, TRAIN_SIZE, VAL_SIZE, TEST_SIZE,
)


def load_data():
    df = pd.read_csv(CLEAN_REVIEWS_PATH)
    print(f"Loaded cleaned dataset with shape: {df.shape}")
    return df


def stratified_sample(df, sample_size=SAMPLE_SIZE, random_state=RANDOM_STATE):
    """
    Draws a stratified sample by Score, preserving the dataset's natural
    rating distribution. If the cleaned dataset is already smaller than
    the requested sample size, the full dataset is used.
    """
    print("\n" + "*" * 50)
    print("STRATIFIED SAMPLING")
    print("*" * 50)

    if len(df) <= sample_size:
        print(f"Dataset ({len(df)} rows) is smaller than the requested sample "
              f"size ({sample_size}); using the full dataset.")
        return df.reset_index(drop=True)

    frac = sample_size / len(df)
    sampled_parts = [
        group.sample(frac=frac, random_state=random_state)
        for _, group in df.groupby("Score")
    ]
    sampled_df = pd.concat(sampled_parts)
    sampled_df = sampled_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"Sampled dataset shape: {sampled_df.shape}")
    print("\nRating distribution (original):")
    print(df["Score"].value_counts(normalize=True).sort_index())
    print("\nRating distribution (sample):")
    print(sampled_df["Score"].value_counts(normalize=True).sort_index())

    return sampled_df


def assign_split(df, train_size=TRAIN_SIZE, val_size=VAL_SIZE, test_size=TEST_SIZE,
                  random_state=RANDOM_STATE):
    """
    Assigns each row to "train", "val", or "test", stratified by Score,
    and stores this assignment as a column so every downstream script
    can filter on it instead of re-splitting independently.

    The validation set exists specifically for hyperparameter selection
    (see 06_svd_model.py and 07_hybrid_model.py); it is never used to
    compute the metrics reported as final results -- those always come
    from the untouched test set. Splitting is done in two stratified
    steps: first test is carved off, then the remainder is split into
    train and validation, which together give exactly the configured
    train_size / val_size / test_size proportions of the full sample.
    """
    print("\n" + "*" * 50)
    print("ASSIGNING CANONICAL TRAIN/VALIDATION/TEST SPLIT")
    print("*" * 50)
    assert abs(train_size + val_size + test_size - 1.0) < 1e-9, \
        "train_size + val_size + test_size must sum to 1.0"

    train_val_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=df["Score"],
    )

    # val_size was expressed as a fraction of the FULL dataset; convert
    # it to a fraction of the remaining train_val_idx pool.
    val_fraction_of_remainder = val_size / (train_size + val_size)

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_fraction_of_remainder,
        random_state=random_state,
        stratify=df.loc[train_val_idx, "Score"],
    )

    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    print(f"Train rows: {(df['split'] == 'train').sum()}")
    print(f"Val rows:   {(df['split'] == 'val').sum()}")
    print(f"Test rows:  {(df['split'] == 'test').sum()}")
    print("\nRating distribution by split:")
    print(df.groupby("split")["Score"].value_counts(normalize=True).sort_index())

    return df


def save_data(df):
    df.to_csv(SPLIT_REVIEWS_PATH, index=False)
    print(f"\nSaved sampled dataset with canonical split to {SPLIT_REVIEWS_PATH}")


if __name__ == "__main__":
    df = load_data()
    df = stratified_sample(df)
    df = assign_split(df)
    save_data(df)