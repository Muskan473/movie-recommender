import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge datasets on 'title'
movies = movies.merge(credits, on='title')

# Select required columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Define safe extraction functions
def extract_list(obj):
    if pd.isna(obj):
        return []
    try:
        return [i['name'] for i in ast.literal_eval(obj)]
    except:
        return []

def extract_cast(obj):
    if pd.isna(obj):
        return []
    try:
        cast = []
        for i in ast.literal_eval(obj):
            if len(cast) < 3:
                cast.append(i['name'])
            else:
                break
        return cast
    except:
        return []

def extract_director(obj):
    if pd.isna(obj):
        return []
    try:
        crew_list = ast.literal_eval(obj)
        for i in crew_list:
            if i['job'] == 'Director':
                return [i['name']]
        return []
    except:
        return []

# Apply the functions
movies['genres'] = movies['genres'].apply(extract_list)
movies['keywords'] = movies['keywords'].apply(extract_list)
movies['cast'] = movies['cast'].apply(extract_cast)
movies['director'] = movies['crew'].apply(extract_director)

# Preprocess overview
movies['overview'] = movies['overview'].fillna('').apply(lambda x: x.split())

print("Using director not crew in tags!")   # Add this to confirm you’re running the correct file
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['director']

movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# Compute cosine similarity
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):
    movie = movie.lower()
    try:
        movie_index = movies[movies['title'].str.lower() == movie].index[0]
    except IndexError:
        print("Movie not found in the database. Try another one.")
        return
    
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    print(f"\nMovies similar to '{movies.iloc[movie_index]['title']}':")
    for i in movie_list:
        print(f"- {movies.iloc[i[0]]['title']}")

# Main loop
if __name__ == "__main__":
    while True:
        movie_name = input("Enter a movie name to get recommendations (or type 'exit' to quit): ")
        if movie_name.lower() == 'exit':
            break
        recommend(movie_name)
