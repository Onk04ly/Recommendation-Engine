import numpy as np
import pandas as pd
from cbf_model import cbf_recommend_for_user


class HybridRecommender:
    """
    Weighted hybrid: hybrid_score = alpha * cf_score_norm + (1 - alpha) * cbf_score_norm

    Both score sets are min-max normalised to [0, 1] before combining so that
    CF's 0.5–5.0 rating scale doesn't dominate CBF's 0–1 cosine scale.
    """

    def __init__(self, svd_model, tfidf_matrix, movies_df, ratings_df, alpha=0.7):
        self.svd = svd_model
        self.tfidf_matrix = tfidf_matrix
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.alpha = alpha
        self._all_movie_ids = set(ratings_df['movieId'].unique())
        self._popular = (
            ratings_df.groupby('movieId').size()
            .sort_values(ascending=False)
            .index.tolist()
        )

    def recommend(self, user_id, n=10):
        user_seen = set(
            self.ratings_df[self.ratings_df['userId'] == user_id]['movieId']
        )
        n_train_ratings = len(
            self.ratings_df[self.ratings_df['userId'] == user_id]
        )

        # Cold-start fallback
        if n_train_ratings == 0:
            return self._popular_fallback(user_id, n)
        if n_train_ratings < 5:
            result = cbf_recommend_for_user(
                user_id, self.ratings_df, self.tfidf_matrix, self.movies_df, top_n=n
            )
            return result.rename(columns={'score': 'hybrid_score'})

        candidates = [m for m in self._all_movie_ids if m not in user_seen]

        # CF scores
        cf_raw = np.array([self.svd.predict(user_id, m).est for m in candidates])

        # CBF scores — get a wide set then map to candidate list
        cbf_df = cbf_recommend_for_user(
            user_id, self.ratings_df, self.tfidf_matrix, self.movies_df,
            top_n=len(candidates), candidates_per_item=50
        )
        cbf_map = dict(zip(cbf_df['movieId'], cbf_df['score']))
        cbf_raw = np.array([cbf_map.get(m, 0.0) for m in candidates])

        cf_norm = _minmax(cf_raw)
        cbf_norm = _minmax(cbf_raw)

        hybrid = self.alpha * cf_norm + (1 - self.alpha) * cbf_norm
        top_idx = np.argsort(hybrid)[::-1][:n]

        return pd.DataFrame({
            'movieId': [candidates[i] for i in top_idx],
            'hybrid_score': hybrid[top_idx],
            'cf_score': cf_raw[top_idx],
            'cbf_score': cbf_raw[top_idx],
        })

    def _popular_fallback(self, user_id, n):
        seen = set(self.ratings_df[self.ratings_df['userId'] == user_id]['movieId'])
        recs = [m for m in self._popular if m not in seen][:n]
        return pd.DataFrame({'movieId': recs, 'hybrid_score': [np.nan] * len(recs)})


def _minmax(arr):
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)
