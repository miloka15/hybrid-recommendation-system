"""
configure.py
Project: Hybrid Recommendation System

Central configuration for the entire pipeline. Every script under src/
imports its paths and constants from here instead of redefining them,
so there is exactly one place to change a path, a sample size, or a
random seed.

Critically, this file also defines the SINGLE canonical sample size,
random state, and test size used to build the train/test split in
02_sampling_and_split.py. Every downstream script (SVD baseline, hybrid
model, evaluation) reuses that same split rather than sampling or
splitting independently. This guarantees the SVD baseline and the
hybrid model are always compared on identical, genuinely held-out data.
"""

import os

# ---------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------
DATA_RAW_DIR = os.path.join("data", "raw")
DATA_PROCESSED_DIR = os.path.join("data", "processed")
RESULTS_DIR = "results"
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

for _dir in (DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, TABLES_DIR,
             FIGURES_DIR, METRICS_DIR):
    os.makedirs(_dir, exist_ok=True)

# ---------------------------------------------------------------------
# Data files (pipeline stages, in order)
# ---------------------------------------------------------------------
RAW_REVIEWS_PATH = os.path.join(DATA_RAW_DIR, "reviews.csv")
CLEAN_REVIEWS_PATH = os.path.join(DATA_PROCESSED_DIR, "clean_reviews.csv")
SPLIT_REVIEWS_PATH = os.path.join(DATA_PROCESSED_DIR, "sampled_reviews_split.csv")
CLEANED_TEXT_PATH = os.path.join(DATA_PROCESSED_DIR, "cleaned_reviews_text.csv")
SENTIMENT_PATH = os.path.join(DATA_PROCESSED_DIR, "reviews_with_sentiment.csv")

# ---------------------------------------------------------------------
# Model / feature artifacts
# ---------------------------------------------------------------------
TFIDF_MATRIX_PATH = os.path.join(MODELS_DIR, "tfidf_matrix.pkl")
TFIDF_VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
TFIDF_REDUCER_PATH = os.path.join(MODELS_DIR, "tfidf_svd_reducer.pkl")
SVD_MODEL_PATH = os.path.join(MODELS_DIR, "svd_baseline.pkl")
HYBRID_MODEL_PATH = os.path.join(MODELS_DIR, "hybrid_model.pkl")
HYBRID_COLDSTART_MODEL_PATH = os.path.join(MODELS_DIR, "hybrid_model_coldstart.pkl")

# ---------------------------------------------------------------------
# Sampling / reproducibility (the single source of truth for the split)
# ---------------------------------------------------------------------
SAMPLE_SIZE = 100000
RANDOM_STATE = 42

# Three-way split: 70% train / 15% validation / 15% test.
# The validation set is used ONLY for hyperparameter selection (Section
# on tuning in 06_svd_model.py and 07_hybrid_model.py); final reported
# metrics always come from the untouched test set.
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# ---------------------------------------------------------------------
# Hyperparameter search grids (small, deliberately -- this project
# prioritises a defensible, disclosed tuning process over an exhaustive
# search; each grid is evaluated on the validation split only)
# ---------------------------------------------------------------------
SVD_PARAM_GRID = [
    {"n_factors": 50, "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02},
    {"n_factors": 100, "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02},
    {"n_factors": 100, "n_epochs": 30, "lr_all": 0.005, "reg_all": 0.05},
    {"n_factors": 50, "n_epochs": 30, "lr_all": 0.01, "reg_all": 0.05},
]

RF_PARAM_GRID = [
    {"n_estimators": 100, "max_depth": None},
    {"n_estimators": 200, "max_depth": None},
    {"n_estimators": 100, "max_depth": 20},
    {"n_estimators": 200, "max_depth": 20},
]

# ---------------------------------------------------------------------
# Sparsity filtering (Objective 1: filter sparse user-item interactions)
# ---------------------------------------------------------------------
MIN_USER_INTERACTIONS = 3   # drop users with fewer than this many reviews
MIN_ITEM_INTERACTIONS = 3   # drop products with fewer than this many reviews
MAX_FILTER_ITERATIONS = 5   # filtering can cascade; cap the number of passes

# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------
N_TFIDF_FEATURES = 5000
N_TFIDF_SVD_COMPONENTS = 50

# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
# Two Precision@K measurements are reported, since sparse review data
# means most users have very few test-set items:
#   PRECISION_AT_K_SMALL: a small K, computed across ALL users with at
#     least one relevant test item (maximum coverage, but for users with
#     fewer than K items "top-K" is really "all their items").
#   PRECISION_AT_K: a larger, more conventional K, computed ONLY on the
#     subset of users who have at least K test items -- a stricter test
#     of genuine top-K ranking quality. The subset's coverage (as a % of
#     all users) is reported alongside it.
PRECISION_AT_K_SMALL = 3
PRECISION_AT_K = 10
RELEVANCE_THRESHOLD = 4  # ratings >= this are treated as "relevant" for Precision@K