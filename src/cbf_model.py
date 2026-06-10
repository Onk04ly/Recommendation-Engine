import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_tfidf_matrix(movies_df, max_features=5000):
    """Fit TF-IDF on the 'content' column (genres + tags). Returns (matrix, vectorizer)."""
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
    )
    matrix = tfidf.fit_transform(movies_df['content'].fillna(''))
    return matrix, tfidf


def get_similar_items(movie_id, tfidf_matrix, movies_df, top_n=20):
    """
    Return top_n most similar movies to movie_id by TF-IDF cosine similarity.
    Computes similarity for one row at a time — avoids materialising the full N×N matrix.
    """
    idx_series = movies_df.index[movies_df['movieId'] == movie_id]
    if idx_series.empty:
        return pd.DataFrame(columns=['movieId', 'similarity'])

    idx = idx_series[0]
    row_vec = tfidf_matrix[idx]
    sims = cosine_similarity(row_vec, tfidf_matrix).flatten()
    sims[idx] = 0.0  # exclude self

    top_indices = np.argpartition(sims, -top_n)[-top_n:]
    top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

    return pd.DataFrame({
        'movieId': movies_df.iloc[top_indices]['movieId'].values,
        'similarity': sims[top_indices],
    })


def cbf_recommend_for_user(user_id, ratings_df, tfidf_matrix, movies_df,
                            top_n=10, liked_threshold=3.5, candidates_per_item=20):
    """
    Content-based recommendations for a user.

    For each item the user liked (rating >= liked_threshold), fetch the top
    candidates_per_item similar items. Aggregate similarity scores by summing
    across source items. Exclude already-rated items. Return top_n.
    """
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    liked = user_ratings[user_ratings['rating'] >= liked_threshold]['movieId'].tolist()

    if not liked:
        liked = user_ratings.nlargest(5, 'rating')['movieId'].tolist()
    if not liked:
        return pd.DataFrame(columns=['movieId', 'score'])

    seen = set(user_ratings['movieId'].tolist())
    scores = {}

    for mid in liked:
        similar = get_similar_items(mid, tfidf_matrix, movies_df, top_n=candidates_per_item)
        for _, row in similar.iterrows():
            cand_id = row['movieId']
            if cand_id in seen:
                continue
            scores[cand_id] = scores.get(cand_id, 0.0) + row['similarity']

    if not scores:
        return pd.DataFrame(columns=['movieId', 'score'])

    result = (
        pd.DataFrame(list(scores.items()), columns=['movieId', 'score'])
        .sort_values('score', ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return result
