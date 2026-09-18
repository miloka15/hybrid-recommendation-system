"""
File: 05_tfidf.py
Project: Hybrid Recommendation System

Purpose:
Fits a TF-IDF vectorizer and transforms the cleaned review text into a
sparse feature matrix.

Note on leakage: the vectorizer's vocabulary and IDF weights are fit
ONLY on the training rows (using the canonical "split" column from
02_sampling_and_split.py), then used to transform both train and test
rows. Fitting on the full dataset (including test rows) would let
vocabulary/IDF statistics derived from test-set reviews influence the
features used to predict those same test rows -- a subtle form of
leakage that is avoided here.
"""

import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

from configure import (
    SENTIMENT_PATH, TFIDF_MATRIX_PATH, TFIDF_VECTORIZER_PATH,
    N_TFIDF_FEATURES,
)


def load_data():
    df = pd.read_csv(SENTIMENT_PATH)
    print(f"Loaded dataset with shape: {df.shape}")
    return df


def compute_tfidf(df, max_features=N_TFIDF_FEATURES):
    print("\n" + "*" * 50)
    print("COMPUTING TF-IDF FEATURES")
    print("*" * 50)

    df["CleanCombined"] = df["CleanCombined"].fillna("")

    train_mask = df["split"] == "train"
    print(f"Fitting TF-IDF vocabulary on {train_mask.sum()} training rows only "
          f"(test rows are transformed, not fit).")

    vectorizer = TfidfVectorizer(max_features=max_features)
    vectorizer.fit(df.loc[train_mask, "CleanCombined"])

    # Transform the FULL dataset (train + test) with the train-fit vectorizer,
    # so the matrix stays row-aligned with reviews_with_sentiment.csv for
    # downstream slicing.
    tfidf_matrix = vectorizer.transform(df["CleanCombined"])

    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Sample feature names: {vectorizer.get_feature_names_out()[:20]}")

    return tfidf_matrix, vectorizer


def save_outputs(tfidf_matrix, vectorizer):
    with open(TFIDF_MATRIX_PATH, "wb") as f:
        pickle.dump(tfidf_matrix, f)
    print(f"Saved TF-IDF matrix to {TFIDF_MATRIX_PATH}")

    with open(TFIDF_VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"Saved TF-IDF vectorizer to {TFIDF_VECTORIZER_PATH}")


if __name__ == "__main__":
    df = load_data()
    tfidf_matrix, vectorizer = compute_tfidf(df)
    save_outputs(tfidf_matrix, vectorizer)