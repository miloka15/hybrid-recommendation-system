"""
File: 01_preprocessing.py
Project: Hybrid Recommendation System

Purpose:
Loads the raw reviews dataset, inspects it, ACTUALLY cleans it
(drops missing values and duplicates, rather than only reporting them),
filters sparse user/item interactions, performs exploratory data
analysis, and saves a cleaned dataset for the rest of the pipeline.

Covers Proposal Objective 1 (preprocessing) and Objective 2 (EDA).
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configure import (
    RAW_REVIEWS_PATH, CLEAN_REVIEWS_PATH, FIGURES_DIR,
    MIN_USER_INTERACTIONS, MIN_ITEM_INTERACTIONS, MAX_FILTER_ITERATIONS,
)

REQUIRED_COLUMNS = ["ProductId", "UserId", "Score", "Summary", "Text"]


def load_data(file_path):
    """
    Loads the dataset from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        print("*" * 50)
        print("Dataset loaded successfully!")
        print("*" * 50)
        return df
    except FileNotFoundError:
        print(f"Error: File not found -> {file_path}")
        return None


def inspect_data(df):
    """
    Displays basic information about the dataset.
    """
    print("\n" + "*" * 50)
    print("Dataset Overview")
    print("*" * 50)
    print(f"\nDataset Shape: {df.shape}")
    print("\nColumn Names:")
    print(df.columns.tolist())
    print("\nData Types:")
    print(df.dtypes)
    print("\nFirst Five Rows:")
    print(df.head())


def check_missing_values(df):
    """
    Reports the number of missing values in each column (diagnostic only;
    actual handling happens in clean_data).
    """
    print("\n" + "*" * 50)
    print("Missing Values (before cleaning)")
    print("*" * 50)
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")
    return missing


def check_duplicates(df):
    """
    Reports the number of fully duplicate rows (diagnostic only; actual
    handling happens in clean_data).
    """
    print("\n" + "*" * 50)
    print("Duplicate Records (before cleaning)")
    print("*" * 50)
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")
    return duplicates


def clean_data(df):
    """
    Selects the columns required for the recommendation system, then
    actually removes missing values and duplicates (rather than only
    reporting them, as the previous version of this script did).
    """
    print("\n" + "*" * 50)
    print("Cleaning Dataset")
    print("*" * 50)

    df = df[REQUIRED_COLUMNS].copy()
    n_before = len(df)

    # Drop rows missing any field essential to the pipeline. UserId/ProductId
    # are required for collaborative filtering; Text is required for the
    # NLP/sentiment side. Summary is allowed to be empty (some reviews have
    # no summary) and is filled with an empty string rather than dropped.
    df["Summary"] = df["Summary"].fillna("")
    df = df.dropna(subset=["UserId", "ProductId", "Score", "Text"])
    n_after_na = len(df)
    print(f"Dropped {n_before - n_after_na} rows with missing UserId/ProductId/Score/Text.")

    df = df.drop_duplicates()
    n_after_dupes = len(df)
    print(f"Dropped {n_after_na - n_after_dupes} exact duplicate rows.")

    df = df.reset_index(drop=True)
    print(f"\nDataset shape after missing-value and duplicate removal: {df.shape}")

    return df


def filter_sparse_interactions(df, min_user=MIN_USER_INTERACTIONS,
                                min_item=MIN_ITEM_INTERACTIONS,
                                max_iterations=MAX_FILTER_ITERATIONS):
    """
    Iteratively removes users and products with fewer than the configured
    minimum number of interactions. This is done iteratively (rather than
    in a single pass) because removing sparse users can push some products
    below the item threshold, and vice versa; a single pass would leave
    residual sparsity.
    """
    print("\n" + "*" * 50)
    print("FILTERING SPARSE USER/ITEM INTERACTIONS")
    print("*" * 50)
    print(f"Thresholds: min {min_user} reviews/user, min {min_item} reviews/product")

    n_before = len(df)

    for i in range(max_iterations):
        user_counts = df["UserId"].value_counts()
        item_counts = df["ProductId"].value_counts()

        valid_users = user_counts[user_counts >= min_user].index
        valid_items = item_counts[item_counts >= min_item].index

        new_df = df[df["UserId"].isin(valid_users) & df["ProductId"].isin(valid_items)]

        print(f"  Iteration {i + 1}: {len(df)} -> {len(new_df)} rows")

        if len(new_df) == len(df):
            df = new_df
            break
        df = new_df

    df = df.reset_index(drop=True)
    print(f"\nRows removed by sparsity filtering: {n_before - len(df)}")
    print(f"Dataset shape after sparsity filtering: {df.shape}")

    return df


def exploratory_data_analysis(df):
    """
    Performs exploratory data analysis covering rating distribution,
    review length, user activity, and product/item activity, and saves
    figures for each. Satisfies Proposal Objective 2 in full (the
    previous version of this script only covered rating distribution
    and review length).
    """
    print("\n" + "=" * 50)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 50)

    # --- Rating distribution ---
    rating_counts = df["Score"].value_counts().sort_index()
    print("\nRating Distribution:")
    print(rating_counts)

    plt.figure(figsize=(8, 5))
    plt.bar(rating_counts.index.astype(str), rating_counts.values, color="#4C72B0")
    plt.title("Distribution of Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/rating_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- Review length ---
    df["ReviewLength"] = df["Text"].apply(lambda x: len(str(x).split()))
    print("\nReview Length (words) Summary:")
    print(df["ReviewLength"].describe())

    plt.figure(figsize=(8, 5))
    plt.hist(df["ReviewLength"], bins=50, color="#4C72B0")
    plt.title("Distribution of Review Length")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/review_length_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- User activity (reviews per user) ---
    user_activity = df["UserId"].value_counts()
    print("\nUser Activity (reviews per user) Summary:")
    print(user_activity.describe())

    plt.figure(figsize=(8, 5))
    plt.hist(user_activity, bins=50, color="#55A868")
    plt.title("Distribution of Reviews per User")
    plt.xlabel("Number of Reviews")
    plt.ylabel("Number of Users")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/user_activity_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- Product / item activity (reviews per product) ---
    item_activity = df["ProductId"].value_counts()
    print("\nProduct Activity (reviews per product) Summary:")
    print(item_activity.describe())

    plt.figure(figsize=(8, 5))
    plt.hist(item_activity, bins=50, color="#C44E52")
    plt.title("Distribution of Reviews per Product")
    plt.xlabel("Number of Reviews")
    plt.ylabel("Number of Products")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/item_activity_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nEDA figures saved to {FIGURES_DIR}/")


def save_data(df):
    """
    Saves the cleaned, filtered dataset.
    """
    df.to_csv(CLEAN_REVIEWS_PATH, index=False)
    print("\nCleaned dataset saved successfully!")
    print(f"Location: {CLEAN_REVIEWS_PATH}")


if __name__ == "__main__":
    df = load_data(RAW_REVIEWS_PATH)

    if df is not None:
        inspect_data(df)
        check_missing_values(df)
        check_duplicates(df)

        df = clean_data(df)
        df = filter_sparse_interactions(df)

        exploratory_data_analysis(df)

        save_data(df)