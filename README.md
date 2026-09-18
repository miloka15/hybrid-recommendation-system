# Hybrid Recommendation System

**COM748 MSc Research Project - Miloka Rebello, University of Ulster**

A hybrid recommendation system combining SVD-based collaborative filtering with review-text features (VADER sentiment and TF-IDF content) through a Random Forest meta-learner, evaluated on the Amazon Fine Food Reviews dataset.

This repository accompanies the Research Paper *"A Hybrid Recommendation System Combining Collaborative Filtering, Sentiment Analysis, and Content-Based Features for Review-Based Rating Prediction"* and the associated Supporting Digital Material.

---

## What this project does

- Trains an SVD collaborative-filtering baseline and a hybrid Random Forest model on an identical, canonical train/validation/test split, avoiding evaluation leakage between models.
- Tunes both models' hyperparameters via a validation-set grid search, then evaluates once on a held-out test set.
- Evaluates two hybrid feature-construction strategies:
  - **Same-review**: sentiment/content features come from the review being predicted.
  - **Leave-one-out (cold-start-safe)**: sentiment/content features come from a product's *other* reviews, simulating information genuinely available before a rating is made.
- Reports RMSE, MAE, Precision@K, per-rating-class error, and Random Forest feature importances for both configurations.

---

## Requirements

- Python 3.10+
- See `requirements.txt` for package versions.


pip install -r requirements.txt


The first run of `03_text_preprocessing.py` will download required NLTK resources (`punkt`, `stopwords`, `wordnet`, `omw-1.4`) automatically — this requires an internet connection the first time only.

---

## Project structure


hybrid-recommendation-system/
│
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   │   └── reviews.csv                  # place the raw Amazon Fine Food Reviews CSV here
│   └── processed/                        # intermediate outputs, created automatically
│
├── src/
│   ├── configure.py                      # central config: paths, seeds, hyperparameter grids
│   ├── 01_preprocessing.py               # cleaning, sparsity filtering, EDA
│   ├── 02_sampling_and_split.py          # canonical stratified sample + 70/15/15 split
│   ├── 03_text_preprocessing.py          # text cleaning (tokenise, lemmatise, stopwords)
│   ├── 04_sentiment_analysis.py          # VADER sentiment scoring
│   ├── 05_tfidf.py                       # TF-IDF vectorisation (fit on train only)
│   ├── 06_svd_model.py                   # SVD baseline: tuning + final evaluation
│   ├── 07_hybrid_model.py                # hybrid model (same-review): tuning + evaluation
│   ├── 07b_hybrid_model_coldstart.py     # hybrid model (leave-one-out / cold-start-safe)
│   ├── 08_evaluation.py                  # diagnostic figures/tables for the hybrid model
│   └── 08b_evaluation_coldstart.py       # diagnostic figures/tables for the cold-start variant
│
└── results/
    ├── models/                           # saved .pkl models (SVD, hybrid, TF-IDF vectorizer)
    ├── tables/                           # CSVs: predictions, tuning results, metrics by rating
    ├── metrics/                          # JSON: headline RMSE/MAE/Precision@K
    └── figures/                          # all PNG figures


---

## Dataset

This project uses the **Amazon Fine Food Reviews** dataset (568,454 reviews), originally released by Stanford's SNAP research group. Download it from Kaggle or the original SNAP source and place the CSV at:


data/raw/reviews.csv

Expected columns: `Id`, `ProductId`, `UserId`, `ProfileName`, `HelpfulnessNumerator`, `HelpfulnessDenominator`, `Score`, `Time`, `Summary`, `Text`.

---

## How to run

Run the scripts **in order** from the project root. Each stage writes its output to `data/processed/` or `results/`, which the next stage reads.


python src/01_preprocessing.py
python src/02_sampling_and_split.py
python src/03_text_preprocessing.py
python src/04_sentiment_analysis.py
python src/05_tfidf.py
python src/06_svd_model.py
python src/07_hybrid_model.py
python src/07B_hybrid_model_coldstart.py
python src/08_evaluation.py
python src/08B_evaluation_coldstart.py
```
**Note:** `07b`, `08`, and `08b` all depend on outputs from earlier stages (`05_tfidf.py` and `06_svd_model.py` in particular) - do not skip steps or run them out of order.

If you only change something in '06_svd_model.py' or later (e.g. re-tuning hyperparameters), you do not need to re-run '01''05', since the split, cleaned text, sentiment scores, and TF-IDF matrix are unaffected.

---

## Key results (from the reported run)

| Model | RMSE | MAE |
|---|---|---|
| SVD-only baseline (tuned) | 1.0068 | 0.7356 |
| Hybrid — same-review | 0.9604 (−4.61%) | 0.5907 (−19.71%) |
| Hybrid — leave-one-out (cold-start-safe) | 0.9998 (−0.69%) | 0.6224 (−15.40%) |

Full results, per-rating-class breakdowns, feature importances, and Precision@K are in `results/tables/` and `results/metrics/` after running the pipeline, and are discussed in full in the accompanying Research Paper (Section IV) and Supporting Digital Material.

---

## Methodological notes

Three corrections were made during development, each documented and quantified rather than silently applied:

1. **Evaluation leakage fix** — a single canonical train/validation/test split (`02_sampling_and_split.py`) is now shared by every model, replacing an earlier version where the SVD baseline and hybrid model each drew independent samples/splits.
2. **Fair hyperparameter tuning** — both the SVD baseline and the Random Forest are tuned via the same validation-set grid search procedure (Section III-H of the paper), rather than only tuning the hybrid model.
3. **Leakage-free cold-start variant** — `07B_hybrid_model_coldstart.py` re-evaluates the hybrid approach using leave-one-out product-level content aggregates instead of same-review information, to test how much of the hybrid model's advantage depends on information unavailable at genuine recommendation time.

See the Research Paper (Sections III–VI) and Supporting Material (Sections 2 and 4) for full discussion.
