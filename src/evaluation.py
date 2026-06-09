import random
from collections import defaultdict


def precision_recall_at_k(predictions, k=10, threshold=4.0):
    """
    Compute average Precision@K and Recall@K from Surprise Prediction objects.

    predictions : list of surprise.Prediction — each has .uid, .iid, .r_ui (true), .est (predicted)
    threshold   : minimum true rating to consider an item 'relevant'
    """
    user_est_true = defaultdict(list)
    for pred in predictions:
        user_est_true[pred.uid].append((pred.est, pred.r_ui))

    precisions, recalls = {}, {}
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        n_rel           = sum(1 for (_, r) in user_ratings if r >= threshold)
        n_rel_and_rec_k = sum(
            1 for (est, r) in user_ratings[:k]
            if r >= threshold
        )

        precisions[uid] = n_rel_and_rec_k / k if k != 0 else 0
        recalls[uid]    = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    avg_p = sum(precisions.values()) / len(precisions)
    avg_r = sum(recalls.values()) / len(recalls)
    return avg_p, avg_r


def hit_rate_at_k(model, train_df, test_df, n_neg=99, k=10, threshold=4.0, seed=42):
    """
    Sampled Hit Rate@K: for each eligible user, rank the test item against
    n_neg random unrated negatives (100 items total). A 'hit' = test item in top-K.

    Only evaluates users whose held-out test item has true rating >= threshold.
    This is the standard evaluation protocol used in most RecSys papers.
    """
    rng = random.Random(seed)
    all_items = set(train_df['movieId'].unique())

    hits, n_eligible = 0, 0

    for _, row in test_df.iterrows():
        uid, test_mid, true_r = row['userId'], row['movieId'], row['rating']
        if true_r < threshold:
            continue

        user_seen  = set(train_df[train_df['userId'] == uid]['movieId'])
        candidates = list(all_items - user_seen - {test_mid})
        negatives  = rng.sample(candidates, min(n_neg, len(candidates)))

        scored = [(model.predict(uid, mid).est, mid) for mid in negatives + [test_mid]]
        scored.sort(reverse=True)

        top_k = [mid for _, mid in scored[:k]]
        if test_mid in top_k:
            hits += 1
        n_eligible += 1

    return hits / n_eligible if n_eligible > 0 else 0.0
