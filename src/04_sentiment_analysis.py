"""
File: 04_sentiment_analysis.py
Project: Hybrid Recommendation System

Purpose:
Computes VADER sentiment scores for each review's raw text, validates
that sentiment tracks star rating as a sanity check, and visualizes the
relationship (completing the "sentiment distribution" part of Proposal
Objective 2, EDA).
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from configure import CLEANED_TEXT_PATH, SENTIMENT_PATH, FIGURES_DIR


def load_data():
    df = pd.read_csv(CLEANED_TEXT_PATH)
    print(f"Loaded dataset with shape: {df.shape}")
    return df


def combine_raw_text(df):
    """
    Combines the raw Summary and Text columns for VADER, deliberately
    using raw (unprocessed) text since VADER relies on capitalization,
    punctuation, and exclamation marks as sentiment signals that the
    text-cleaning pipeline in 03_text_preprocessing.py removes.
    """
    print("\n" + "*" * 50)
    print("COMBINING RAW TEXT FOR VADER")
    print("*" * 50)

    df["Summary"] = df["Summary"].fillna("")
    df["Text"] = df["Text"].fillna("")
    df["RawCombined"] = (df["Summary"] + ". " + df["Text"]).str.strip()

    print("Sample combined text:")
    print(df["RawCombined"].iloc[0][:200])

    return df


def compute_vader_sentiment(df):
    print("\n" + "*" * 50)
    print("COMPUTING VADER SENTIMENT")
    print("*" * 50)

    analyzer = SentimentIntensityAnalyzer()

    def get_scores(text):
        scores = analyzer.polarity_scores(str(text))
        return pd.Series({
            "vader_compound": scores["compound"],
            "vader_pos": scores["pos"],
            "vader_neu": scores["neu"],
            "vader_neg": scores["neg"],
        })

    print("Scoring reviews...")
    sentiment_scores = df["RawCombined"].apply(get_scores)
    df = pd.concat([df, sentiment_scores], axis=1)

    print("\nSentiment score summary:")
    print(df[["vader_compound", "vader_pos", "vader_neu", "vader_neg"]].describe())

    return df


def validate_against_ratings(df):
    """
    Sanity check: mean VADER compound score should increase with star
    rating if sentiment is being captured correctly. Also saved as a
    figure for the EDA section of the dissertation.
    """
    print("\n" + "*" * 50)
    print("VALIDATION: MEAN VADER COMPOUND SCORE BY STAR RATING")
    print("*" * 50)

    means = df.groupby("Score")["vader_compound"].mean()
    print(means)

    plt.figure(figsize=(8, 5))
    df.boxplot(column="vader_compound", by="Score", grid=False)
    plt.title("VADER Compound Sentiment by Star Rating")
    plt.suptitle("")
    plt.xlabel("Star Rating")
    plt.ylabel("VADER Compound Score")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/sentiment_by_rating.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES_DIR}/sentiment_by_rating.png")

    return means


def save_data(df):
    df.to_csv(SENTIMENT_PATH, index=False)
    print(f"\nSaved dataset with sentiment scores to {SENTIMENT_PATH}")


if __name__ == "__main__":
    df = load_data()
    df = combine_raw_text(df)
    df = compute_vader_sentiment(df)
    validate_against_ratings(df)
    save_data(df)