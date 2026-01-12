from put_me_on_model import extract_test_data
from put_me_on_model import find_movie

matches = []

for _, row in extract_test_data.iterrows():
    movie = find_movie(row["title_norm"], row["Year"])
    matches.append(movie)

extract_test_data["tmdb_match"] = matches

total = len(extract_test_data)
found = extract_test_data["tmdb_match"].notna().sum()
missing = extract_test_data[extract_test_data["tmdb_match"].isna()]

print(f"Matched {found}/{total} movies ({found/total:.1%})")

print("\nMissing movies:")
print(missing[["Name", "Year", "Rating"]].to_string(index=False))