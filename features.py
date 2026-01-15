import pandas as pd
from config import LIKE_THRESHOLD, W_OVERVIEW, W_GENRES, W_DIRECTOR, W_CAST, W_LANG

def add_liked_label(ratings_df):
    ratings_df["Liked"] = (ratings_df["Rating"] >= LIKE_THRESHOLD).astype(int)
    return ratings_df

def expand_tmdb_fields(matched_df):
    # extract the features we need from tmdb_match
    def safe_extract(m):
        if m is None or (isinstance(m, float) and pd.isna(m)):
            return {}
        if isinstance(m, pd.Series):
            return m.to_dict()
        return {}
    
    tmdb_expanded = matched_df["tmdb_match"].apply(safe_extract).apply(pd.Series)

    # pull only the columns you care about (and fill missing safely)
    cols = ["id","title","year","overview","genres","director","cast","original_language","vote_count","vote_average","runtime"]
    tmdb_expanded = tmdb_expanded.reindex(columns=cols)

    out = pd.concat([matched_df.reset_index(drop=True), tmdb_expanded.reset_index(drop=True)], axis=1)

    # fill text fields
    for c in ["overview","genres","director","cast","original_language"]:
        out[c] = out[c].fillna("")

    return out


# combine extracted features into one field: overview, genres, director, cast, and original language
def build_weighted_text(df):
    df["text_features"] = (
        (df["overview"] + " ") * W_OVERVIEW +
        (df["genres"] + " ") * W_GENRES +
        (df["director"] + " ") * W_DIRECTOR +
        (df["cast"] + " ") * W_CAST +
        (df["original_language"] + " ") * W_LANG
    )
    return df