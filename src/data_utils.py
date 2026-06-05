import pandas as pd
import numpy as np


def load_and_clean_ratings(path, min_ratings=5):
    df = pd.read_csv(path)

    df = df.drop_duplicates()

    df = df.sort_values(['userId', 'movieId', 'timestamp'])
    df = df.drop_duplicates(subset=['userId', 'movieId'], keep='last')

    df['rating_date'] = pd.to_datetime(df['timestamp'], unit='s')

    df = df.groupby('userId').filter(lambda x: len(x) >= min_ratings)
    df = df.groupby('movieId').filter(lambda x: len(x) >= min_ratings)

    return df.reset_index(drop=True)


def encode_ids(df):
    user_ids = df['userId'].unique()
    item_ids = df['movieId'].unique()

    user2idx = {uid: idx for idx, uid in enumerate(user_ids)}
    item2idx = {mid: idx for idx, mid in enumerate(item_ids)}

    df = df.copy()
    df['user_idx'] = df['userId'].map(user2idx)
    df['item_idx'] = df['movieId'].map(item2idx)

    return df, user2idx, item2idx


def load_movies_with_tags(movies_path, tags_path):
    movies = pd.read_csv(movies_path)

    movies['genres_str'] = movies['genres'].str.replace('|', ' ', regex=False).str.lower()

    tags = pd.read_csv(tags_path)
    tags['tag'] = tags['tag'].astype(str).str.lower()
    tags_grouped = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    tags_grouped.columns = ['movieId', 'tags_str']

    movies = movies.merge(tags_grouped, on='movieId', how='left')
    movies['tags_str'] = movies['tags_str'].fillna('')

    movies['content'] = movies['genres_str'] + ' ' + movies['tags_str']
    movies['content'] = movies['content'].str.strip()

    return movies
