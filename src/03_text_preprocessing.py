"""
File: 03_text_preprocessing.py
Project: Hybrid Recommendation System

Purpose:
Cleans review text (lowercasing, HTML/URL/punctuation/number removal,
tokenization, stopword removal, lemmatization) for the sampled dataset
produced by 02_sampling_and_split.py. Operating on the sample (rather
than the full ~500k-row dataset, as the original version of this script
did) makes this step considerably faster with no loss of validity,
since only the sampled rows are ever used downstream.

The "split" column from 02_sampling_and_split.py is preserved untouched
so every later script can still identify train vs. test rows.
"""

import pandas as pd
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from configure import SPLIT_REVIEWS_PATH, CLEANED_TEXT_PATH


def download_nltk_resources():
    """
    Downloads required NLTK resources if not already present.
    """
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
    }
    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)


def custom_stopwords():
    """
    Builds a custom stopword list that removes common English stopwords
    but preserves negation terms, which are critical for sentiment
    analysis (e.g. "not good" should not lose its negation).
    """
    stop_words = set(stopwords.words("english"))

    negation_words = {
        "not", "no", "nor", "never", "none", "nobody", "nothing",
        "neither", "nowhere", "cannot", "can't", "won't", "isn't",
        "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't",
        "haven't", "hasn't", "hadn't", "wouldn't", "shouldn't",
        "couldn't", "mustn't"
    }

    custom = stop_words - negation_words
    print(f"Custom stopword list size: {len(custom)} "
          f"(removed {len(negation_words)} negation terms)")

    return custom


def load_data():
    df = pd.read_csv(SPLIT_REVIEWS_PATH)
    print(f"Dataset loaded successfully: {df.shape}")
    return df


def remove_tags(text):
    return re.sub(r"<.*?>", " ", text)


def remove_urls(text):
    return re.sub(r"http\S+|www\S+", " ", text)


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_numbers(text):
    return re.sub(r"\d+", " ", text)


def clean_text(text, stop_words, lemmatizer):
    """
    Full cleaning pipeline for a single piece of text: lowercase ->
    remove HTML -> remove URLs -> remove punctuation -> remove numbers
    -> tokenize -> remove stopwords -> lemmatize.
    """
    text = str(text).lower()
    text = remove_tags(text)
    text = remove_urls(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)

    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]

    return " ".join(tokens)


def preprocess_dataset(df, stop_words, lemmatizer):
    print("\n" + "*" * 50)
    print("TEXT PREPROCESSING")
    print("*" * 50)

    df["Text"] = df["Text"].fillna("")
    df["Summary"] = df["Summary"].fillna("")

    print("Cleaning review text...")
    df["CleanText"] = df["Text"].apply(lambda x: clean_text(x, stop_words, lemmatizer))

    print("Cleaning review summary...")
    df["CleanSummary"] = df["Summary"].apply(lambda x: clean_text(x, stop_words, lemmatizer))

    df["CleanCombined"] = (df["CleanSummary"] + " " + df["CleanText"]).str.strip()

    print("\nSample before/after:")
    print("Original Text:  ", str(df["Text"].iloc[0])[:150])
    print("Cleaned Text:   ", str(df["CleanText"].iloc[0])[:150])

    return df


def save_data(df):
    df.to_csv(CLEANED_TEXT_PATH, index=False)
    print(f"\nSaved preprocessed dataset to {CLEANED_TEXT_PATH}")


if __name__ == "__main__":
    download_nltk_resources()
    stop_words = custom_stopwords()
    lemmatizer = WordNetLemmatizer()

    df = load_data()
    df = preprocess_dataset(df, stop_words, lemmatizer)
    save_data(df)