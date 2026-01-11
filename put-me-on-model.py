import kaggle
import pandas as pd
from xgboost import XGBClassifier

kaggle.api.dataset_download_files('rounakbanik/the-movies-dataset', 
path='./movie_data', 
unzip=True)

movie_ds_metadata = pd.read_csv('movie_data/movies_metadata.csv', low_memory=False)

#loading test data
extract_test_data = pd.read_csv('letterboxd-quang_ak47-2026-01-11-12-28-utc/ratings.csv')
print(extract_test_data.columns.tolist())
print(extract_test_data['Rating'].mean())

#binary liked column (4+ stars)
extract_test_data['Liked'] = (extract_test_data['Rating'] >= 4).astype(int)
# print(f"Movies I liked: {(extract_test_data['Liked'] == 1).sum()}")
# print(f"Mid ahh movies: {(extract_test_data['Liked'] == 0).sum()}")
# print(f"Movies I liked: {extract_test_data[extract_test_data['Liked'] == 1][['Name', 'Rating', 'Liked']].head(10)}")
# print(f"Favourites: {extract_test_data[extract_test_data['Rating'] == 5]}")


movie_ds_metadata['year'] = pd.to_datetime(movie_ds_metadata['release_date'], errors = 'coerce').dt.year
movie_ds_metadata['year'] = movie_ds_metadata['year'].fillna(0).astype(int)
print("All movies from 2019 with 'Parasite' in title:")
parasite_2019 = movie_ds_metadata[(movie_ds_metadata['title'].str.contains('Parasite', case=False, na=False)) & (movie_ds_metadata['year'] == 2019)]
print(parasite_2019[['title', 'year', 'original_title']])
print("\nAll movies with exact title 'Parasite':")
exact_parasite = movie_ds_metadata[movie_ds_metadata['title'] == 'Parasite']
print(exact_parasite[['title', 'year']])

def find_movie(title, year):
    result = movie_ds_metadata[(movie_ds_metadata['title'] == title) & (movie_ds_metadata['year'] == year)]
    if len(result) == 0:
        print("No movie found.")
        return None
    return result.iloc[0]