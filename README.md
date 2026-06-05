# Personalized Recommendation Engine

A hybrid recommendation system that combines collaborative filtering and content-based filtering to deliver personalized item recommendations — built from scratch using Python.

## What This Builds

Most recommendation systems fail in one of two ways: they ignore item content entirely (pure collaborative filtering), or they ignore user behavior entirely (pure content-based). This project builds a **hybrid engine** that uses both signals, weighted intelligently, to produce recommendations that are both personalized and discoverable.

The system is trained and evaluated on the [MovieLens](https://grouplens.org/datasets/movielens/) dataset — a well-studied benchmark with 100K+ explicit ratings from real users, rich genre metadata, and user-generated tags.

## Tech Stack

| Purpose | Library |
|---|---|
| Data wrangling | Pandas, NumPy |
| Collaborative filtering | Scikit-Surprise, Scikit-learn |
| Content-based filtering | Scikit-learn (TF-IDF) |
| Deep learning (optional NCF) | TensorFlow / Keras |
| Visualization | Matplotlib, Seaborn |
| Interactive demo | Streamlit |
| Environment | Python 3.x, Jupyter |

## Project Structure

```
Recommendation-Engine/
├── data/
│   ├── raw/          ← original dataset files (not committed)
│   └── processed/    ← cleaned CSVs and saved visualizations
├── notebooks/        ← Jupyter notebooks
├── src/              ← reusable Python modules
├── app/              ← interactive demo (Streamlit)
├── WEEK_PLAN.md      ← full implementation guide
└── requirements.txt
```
