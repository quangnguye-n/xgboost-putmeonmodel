TMDB_CSV = "new-movie-dataset/TMDB_all_movies.csv"
LETTERBOXD_RATINGS_CSV = "letterboxd-quang_ak47-2026-01-11-12-28-utc/ratings.csv"

LIKE_THRESHOLD = 4.0

TFIDF_MAX_FEATURES = 2000
TFIDF_MIN_DF = 2
TFIDF_STOP_WORDS = "english"
TFIDF_NGRAM_RANGE = (1, 2)

# Field weights (you can tune these)
W_OVERVIEW = 3
W_GENRES = 2
W_DIRECTOR = 1
W_CAST = 1
W_LANG = 1
