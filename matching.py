import re
import pandas as pd

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
def add_title_norm(movie_data, ratings_df):
    movie_data["title_norm"] = movie_data["title"].apply(normalise_title)
    ratings_df["title_norm"] = ratings_df["Name"].apply(normalise_title)
    return movie_data, ratings_df

# method to find if movie exists in TMDB
def find_movie(movie_data, title: str, year: int):
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
        best = result.sort_values("vote_count", ascending=False).iloc[0]
        print(f"Warning: '{title_norm}' found with year {result.iloc[0]['year']} instead of {year}")
        return best

    # if none were found within +/- 2 year search, return most popular movie with that title (most votes on IMDb)
    result = movie_data[movie_data['title_norm'] == title_norm]
    if len(result) > 0:
        best = result.sort_values('vote_count', ascending=False).iloc[0]
        print(f"Warning: '{title_norm}' found but year mismatch: {best['year']} vs {year}")
        print(f"Maybe you were searching for: {best['title']} ({best['year']})")
        return best

    print(f"Error: '{title_norm}' ({year}) not found.")
    return None

def match_ratings_to_tmdb(movie_data, ratings_df):
    # find movies from my letterboxd data in tmdb
    matches = []
    for _, row in ratings_df.iterrows():
        matches.append(find_movie(movie_data, row["Name"], int(row["Year"])))
    ratings_df["tmdb_match"] = matches
    return ratings_df