import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_user_item_matrix(train_df):
    """Pivot train ratings into a userId × movieId DataFrame (NaN = unobserved)."""
    return train_df.pivot_table(index='userId', columns='movieId', values='rating')


def compute_user_similarity(matrix):
    """Cosine similarity between every pair of users (fills NaN with 0 before computing)."""
    filled = matrix.fillna(0).values
    sim = cosine_similarity(filled)
    return pd.DataFrame(sim, index=matrix.index, columns=matrix.index)


def compute_item_similarity(matrix):
    """Cosine similarity between every pair of items (transposes matrix so items are rows)."""
    filled = matrix.fillna(0).values.T  # items × users
    sim = cosine_similarity(filled)
    return pd.DataFrame(sim, index=matrix.columns, columns=matrix.columns)


def predict_user_based(user_id, movie_id, matrix, user_sim, k=20, global_mean=3.5):
    """Mean-centered user-based CF prediction."""
    if user_id not in matrix.index or movie_id not in matrix.columns:
        return global_mean

    item_col = matrix[movie_id].dropna()
    raters = item_col.index.tolist()

    if len(raters) == 0 or user_id not in user_sim.index:
        return global_mean

    sims = user_sim.loc[user_id, raters].drop(labels=[user_id], errors='ignore')
    sims = sims[sims > 0].nlargest(k)

    if sims.empty or sims.sum() == 0:
        return float(matrix.loc[user_id].mean() or global_mean)

    user_mean = matrix.loc[user_id].mean()
    numerator = sum(sim * (item_col[nb] - matrix.loc[nb].mean()) for nb, sim in sims.items())
    denominator = sims.abs().sum()

    return float(np.clip(user_mean + numerator / denominator, 0.5, 5.0))


def predict_item_based(user_id, movie_id, matrix, item_sim, k=20, global_mean=3.5):
    """Item-based CF prediction using weighted average of similar items the user rated."""
    if user_id not in matrix.index or movie_id not in matrix.columns:
        return global_mean

    user_row = matrix.loc[user_id].dropna()
    rated_items = [i for i in user_row.index if i != movie_id]

    if len(rated_items) == 0:
        return global_mean

    sims = item_sim.loc[movie_id, rated_items].nlargest(k)
    sims = sims[sims > 0]

    if sims.empty or sims.sum() == 0:
        return global_mean

    numerator = sum(sim * user_row[item] for item, sim in sims.items())
    denominator = sims.abs().sum()

    return float(np.clip(numerator / denominator, 0.5, 5.0))
