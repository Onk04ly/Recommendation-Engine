# Personalized Recommendation Engine

A hybrid recommendation system that combines collaborative filtering and content-based filtering to deliver personalized item recommendations — built from scratch using Python.

## What This Builds

Most recommendation systems fail in one of two ways: they ignore item content entirely (pure collaborative filtering), or they ignore user behavior entirely (pure content-based). This project builds a **hybrid engine** that uses both signals, weighted intelligently, to produce recommendations that are both personalized and discoverable.

The system is trained and evaluated on the [MovieLens](https://grouplens.org/datasets/movielens/) dataset — a well-studied benchmark with 100K+ explicit ratings from real users, rich genre metadata, and user-generated tags.

## Dataset

**MovieLens ml-latest-small** — sourced from [GroupLens](https://grouplens.org/datasets/movielens/latest/)

| File | Description |
|---|---|
| `ratings.csv` | userId, movieId, rating (0.5–5.0), timestamp |
| `movies.csv` | movieId, title, genres (pipe-separated) |
| `tags.csv` | userId, movieId, user-generated tag, timestamp |
| `links.csv` | movieId, IMDB ID, TMDB ID |

Raw files live in `data/raw/` and are not committed to git (see `.gitignore`).

## Exploratory Data Analysis — Key Findings

After cleaning the dataset (filtered users and items with fewer than 5 ratings), we ran 5 visualizations to understand the data before building any model.

### Dataset Stats After Cleaning

| Metric | Value |
|---|---|
| Users | ~610 |
| Movies (after filter) | ~1,500–2,000 |
| Total Ratings | ~100,000 |
| Sparsity | ~98.3% |

---

### Plot 1 — Rating Distribution

![Rating Distribution](data/processed/viz_rating_dist.png)

Rating 4.0 is the most common (~25K occurrences), followed by 3.0 and 5.0. Ratings below 2.5 are rare. This is **positivity bias** — users mostly rate content they chose to watch and enjoyed. The model's baseline prediction should target ~3.5–4.0, not the scale midpoint of 2.5.

---

### Plot 2 — User Activity Distribution

![User Activity](data/processed/viz_user_activity.png)

Most users gave fewer than 100 ratings. A small number of "super users" gave 1,000–2,100 ratings. The log scale reveals how extreme this skew is. Without normalization, memory-based collaborative filtering would be dominated by these power users.

---

### Plot 3 — Item Popularity (Long Tail)

![Item Popularity](data/processed/viz_item_popularity.png)

Top movies receive 300–400 ratings; beyond rank ~1,000, most items have fewer than 50 ratings. This is the classic **long-tail problem** — collaborative filtering struggles with niche items due to lack of signal. Content-based filtering handles these better since it relies on metadata, not interaction counts.

---

### Plot 4 — Interaction Matrix Sparsity

![Matrix Sparsity](data/processed/viz_matrix_sparsity.png)

Even among the top 50 most active users and top 50 most popular movies, visible white patches (unrated pairs) exist. The full 610×9,000+ matrix is ~98% empty. This confirms why collaborative filtering alone is insufficient — most user pairs share almost no rated movies in common.

---

### Plot 5 — Temporal Rating Trends

![Temporal Trends](data/processed/viz_temporal.png)

Ratings span from 1996 to 2018. Two notable activity spikes: around 2000 and again around 2015–2016. Activity dropped near-zero in 1998–1999 before recovering. The irregular bursts suggest users rate in bulk sessions rather than consistently over time.

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
