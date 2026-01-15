TMDB_CSV = "new-movie-dataset/TMDB_all_movies.csv"
LETTERBOXD_RATINGS_CSV = "letterboxd-quang_ak47-2026-01-11-12-28-utc/ratings.csv"

LIKE_THRESHOLD = 4.0

TFIDF_MAX_FEATURES = 3000
TFIDF_MIN_DF = 2
TFIDF_STOP_WORDS = "english"
TFIDF_NGRAM_RANGE = (1, 3) # include 1, 2, 3-word phrases

# how much weight each column holds in IDF
W_OVERVIEW = 5
W_GENRES = 4
W_DIRECTOR = 3
W_CAST = 2
W_LANG = 1
