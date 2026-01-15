from data_load import load_tmdb, load_letterboxd
from matching import add_title_norm, match_ratings_to_tmdb
from features import add_liked_label, expand_tmdb_fields, build_weighted_text
from tfidf_model import fit_tfidf
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# export dataset from kaggle
# kaggle.api.dataset_download_files('alanvourch/tmdb-movies-daily-updates', 
# path='./new-movie-dataset', 
# unzip=True)

def main():
    # load our datasets
    movie_data = load_tmdb()
    my_ratings = load_letterboxd()

    # normalise titles
    movie_data, my_ratings = add_title_norm(movie_data, my_ratings)

    # add column to indicate movies liked in my_ratings
    my_ratings = add_liked_label(my_ratings)

    # find all the movies from my dataset in tmdb
    my_ratings = match_ratings_to_tmdb(movie_data, my_ratings)

    # isolate matched movies
    matched = my_ratings[my_ratings["tmdb_match"].notna()].copy()

    # concatenate weighted features (overview, genre etc) into new column
    matched = expand_tmdb_fields(matched)
    matched = build_weighted_text(matched)

    # perform tf-idf on our matched movies list's features column
    vectoriser, X = fit_tfidf(matched["text_features"])
    y = matched["Liked"].values

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, 
        matched.index, # pass the original indices
        test_size=0.1, # 20% for testing
        random_state=42, # for reproducibility
        stratify=y # keep same ratio of liked/disliked in both sets
    )

    print("Matched movies:", len(matched))
    print("TF-IDF shape:", X.shape)
    print("Sample features:", vectoriser.get_feature_names_out()[:20])

    print(f"\nTraining set: {X_train.shape[0]} movies")
    print(f"Test set: {X_test.shape[0]} movies")
    print(f"Training - Liked: {y_train.sum()}, Disliked: {len(y_train) - y_train.sum()}")
    print(f"Test - Liked: {y_test.sum()}, Disliked: {len(y_test) - y_test.sum()}")

    # create and train the XGBoost model
    model = XGBClassifier(
        n_estimators=100, # number of trees
        max_depth=3, # keep trees shallow (helps with small dataset)
        learning_rate=0.1, # how fast the model learns
        random_state=42
    )

    # train the model
    print("\nTraining model...")
    model.fit(X_train, y_train)

    # make predictions on test set
    predictions = model.predict(X_test)

    test_movies = matched.loc[test_idx]
    test_movies['predicted'] = predictions
    test_movies['correct'] = test_movies['predicted'] == test_movies['Liked']

    print("\nTest Set Predictions:")
    print(test_movies[['Name', 'Year', 'Rating', 'Liked', 'predicted', 'correct']])

    print(f"\nCorrect predictions: {test_movies['correct'].sum()}/{len(test_movies)}")

if __name__ == "__main__":
    main()