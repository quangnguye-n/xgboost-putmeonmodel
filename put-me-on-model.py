import kaggle
import pandas as pd
from xgboost import XGBClassifier


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

def find_movie(title, year):
    # first try to find movie with exact title and year
    result = movie_data[
        (movie_data['title'] == title) & 
        (movie_data['year'] == year)]
    if len(result) > 0:
        return result.iloc[0]
    
    # if no movie was found from that year, expand search by +/- 2 year
    result = movie_data[
        (movie_data['title'] == title) & 
        (movie_data['year'] >= year - 2) & 
        (movie_data['year'] <= year + 2)]
    if len(result) > 0:
        print(f"Warning: '{title}' found with year {result.iloc[0]['year']} instead of {year}")
        return result.iloc[0]

    # if none were found within +/- 2 year search, return most popular movie with that title (most votes on IMDb)
    result = movie_data[movie_data['title'] == title]
    if len(result) > 0:
        best_match = result.sort_values('vote_count', ascending=False).iloc[0]
        print(f"Warning: '{title}' found but year mismatch: {best_match['year']} vs {year}")
        print(f"Maybe you were searching for: {best_match['title']} ({best_match['year']})")
        return best_match

    print(f"Error: '{title}' ({year}) not found.")
    return None

# #binary liked column (4+ stars)
# extract_test_data['Liked'] = (extract_test_data['Rating'] >= 4).astype(int)
# # print(f"Movies I liked: {(extract_test_data['Liked'] == 1).sum()}")
# # print(f"Mid ahh movies: {(extract_test_data['Liked'] == 0).sum()}")
# # print(f"Movies I liked: {extract_test_data[extract_test_data['Liked'] == 1][['Name', 'Rating', 'Liked']].head(10)}")
# # print(f"Favourites: {extract_test_data[extract_test_data['Rating'] == 5]}")


# movie_ds_metadata['year'] = pd.to_datetime(movie_ds_metadata['release_date'], errors = 'coerce').dt.year
# movie_ds_metadata['year'] = movie_ds_metadata['year'].fillna(0).astype(int)
# print("All movies from 2019 with 'Parasite' in title:")
# parasite_2019 = movie_ds_metadata[(movie_ds_metadata['title'].str.contains('Parasite', case=False, na=False)) & (movie_ds_metadata['year'] == 2019)]
# print(parasite_2019[['title', 'year', 'original_title']])
# print("\nAll movies with exact title 'Parasite':")
# exact_parasite = movie_ds_metadata[movie_ds_metadata['title'] == 'Parasite']
# print(exact_parasite[['title', 'year']])