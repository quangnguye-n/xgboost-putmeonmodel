import pandas as pd
from config import TMDB_CSV, LETTERBOXD_RATINGS_CSV


# load data from kaggle dataset
def load_tmdb():
    movie_data = pd.read_csv(TMDB_CSV, low_memory=False)
    movie_data['year'] = pd.DatetimeIndex(movie_data['release_date']).year # extract year from release_date in dataset
    return movie_data

def load_letterboxd():
    extract_test_data = pd.read_csv(LETTERBOXD_RATINGS_CSV)
    return extract_test_data

