import kaggle
import pandas as pd
from xgboost import XGBClassifier
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# export dataset from kaggle
# kaggle.api.dataset_download_files('alanvourch/tmdb-movies-daily-updates', 
# path='./new-movie-dataset', 
# unzip=True)

# load data from kaggle dataset
movie_data = pd.read_csv('new-movie-dataset/TMDB_all_movies.csv', low_memory=False)

# load test data
extract_test_data = pd.read_csv('letterboxd-quang_ak47-2026-01-11-12-28-utc/ratings.csv')

# extract year from release_date in dataset
movie_data['year'] = pd.DatetimeIndex(movie_data['release_date']).year

# method to normalise movie titles - eg (500) Days of Summer -> 500 Days of Summer
def normalise_title(title):
    if pd.isna(title):
        return ""
    title = title.lower()
    title = re.sub(r"[()]", "", title) # remove only the parentheses characters
    title = re.sub(r"[^a-z0-9\s]", "", title) # remove punctuation
    title = re.sub(r"\s+", " ", title).strip()
    return title

# apply normalisation
movie_data["title_norm"] = movie_data["title"].apply(normalise_title)
extract_test_data["title_norm"] = extract_test_data["Name"].apply(normalise_title)

# method to find if movie exists in TMDB
def find_movie(title, year):
    title_norm = normalise_title(title)
    
    # first try to find movie with exact title and year
    result = movie_data[
        (movie_data['title_norm'] == title_norm) & 
        (movie_data['year'] == year)]
    if len(result) > 0:
        return result.iloc[0]
    
    # if no movie was found from that year, expand search by +/- 2 year
    result = movie_data[
        (movie_data['title_norm'] == title_norm) & 
        (movie_data['year'] >= year - 2) &
        (movie_data['year'] <= year + 2)]
    if len(result) > 0:
        print(f"Warning: '{title_norm}' found with year {result.iloc[0]['year']} instead of {year}")
        return result.iloc[0]

    # if none were found within +/- 2 year search, return most popular movie with that title (most votes on IMDb)
    result = movie_data[movie_data['title_norm'] == title_norm]
    if len(result) > 0:
        best_match = result.sort_values('vote_count', ascending=False).iloc[0]
        print(f"Warning: '{title_norm}' found but year mismatch: {best_match['year']} vs {year}")
        print(f"Maybe you were searching for: {best_match['title']} ({best_match['year']})")
        return best_match

    print(f"Error: '{title_norm}' ({year}) not found.")
    return None

#binary liked column (4+ stars)
extract_test_data['Liked'] = (extract_test_data['Rating'] >= 4).astype(int)

# find movies from my letterboxd data in tmdb
matches = []

for _, row in extract_test_data.iterrows():
    movie = find_movie(row["Name"], row["Year"])
    matches.append(movie)

extract_test_data["tmdb_match"] = matches

total = len(extract_test_data)
found = extract_test_data["tmdb_match"].notna().sum()
missing = extract_test_data[extract_test_data["tmdb_match"].isna()]

# filter to only successfully matched movies
matched_movies = extract_test_data[extract_test_data["tmdb_match"].notna()].copy()

# extract the features we need from tmdb_match
tmdb_expanded = matched_movies["tmdb_match"].apply(
    lambda m: {} if m is None or (isinstance(m, float) and pd.isna(m)) else m.to_dict()
).apply(pd.Series)

# pull only the columns you care about (and fill missing safely)
cols_needed = ["overview", "genres", "director", "cast", "original_language", "id", "title", "year", "vote_count", "vote_average"]
tmdb_expanded = tmdb_expanded.reindex(columns=cols_needed).fillna("")

matched_movies = pd.concat([matched_movies.reset_index(drop=True), tmdb_expanded.reset_index(drop=True)], axis=1)


print(f"Working with {len(matched_movies)} movies")
print(f"Liked: {(matched_movies['Liked'] == 1).sum()}")
print(f"Didn't like: {(matched_movies['Liked'] == 0).sum()}")

missing_overview = matched_movies[matched_movies['overview'] == ""]
print(f"\nMovies with missing overviews: {len(missing_overview)}")
if len(missing_overview) > 0:
    print(missing_overview[['Name', 'Year']])


# combine extracted features into one field: overview, genres, director, cast, and original language
matched_movies["text_features"] = (
    matched_movies["overview"] + " " +
    matched_movies["genres"] + " " +
    matched_movies["director"] + " " +
    matched_movies["cast"] + " " +
    matched_movies["original_language"]
)

# create tf-idf vectoriser
vectoriser = TfidfVectorizer(
    max_features=500, # limits vocab with 500 most important words since we are using small dataset
    min_df=2, # word has to appear in at least two movies
    stop_words='english' # remove common words like 'the', 'a', 'is'
)

# convert text to tf-idf vectors
X = vectoriser.fit_transform(matched_movies['text_features'])
y = matched_movies['Liked'].values

print(f"\nTF-IDF matrix shape: {X.shape}")
print(f"Features: {X.shape[1]} TF-IDF features")
print(f"Samples: {X.shape[0]} movies")

# see some of the important words
feature_names = vectoriser.get_feature_names_out()
print(f"\nSample TF-IDF features: {list(feature_names[:20])}")