from data_load import load_tmdb, load_letterboxd
from matching import add_title_norm, match_ratings_to_tmdb
from features import add_liked_label, expand_tmdb_fields, build_weighted_text
from tfidf_model import fit_tfidf

# export dataset from kaggle
# kaggle.api.dataset_download_files('alanvourch/tmdb-movies-daily-updates', 
# path='./new-movie-dataset', 
# unzip=True)

def main():
    # load our datasets
    movie_data = load_tmdb()
    ratings = load_letterboxd()

    # normalise titles
    movie_data, ratings = add_title_norm(movie_data, ratings)

    ratings = add_liked_label(ratings)
    ratings = match_ratings_to_tmdb(movie_data, ratings)


    matched = ratings[ratings["tmdb_match"].notna()].copy()
    matched = expand_tmdb_fields(matched)
    matched = build_weighted_text(matched)

    vectoriser, X = fit_tfidf(matched["text_features"])
    y = matched["Liked"].values

    print("Matched movies:", len(matched))
    print("TF-IDF shape:", X.shape)
    print("Sample features:", vectoriser.get_feature_names_out()[:20])

if __name__ == "__main__":
    main()