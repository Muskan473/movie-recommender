import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge datasets
movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

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

movies['genres'] = movies['genres'].apply(extract_list)
movies['keywords'] = movies['keywords'].apply(extract_list)
movies['cast'] = movies['cast'].apply(extract_cast)
movies['director'] = movies['crew'].apply(extract_director)

movies['overview'] = movies['overview'].fillna('').apply(lambda x: x.split())
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['director']
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

def recommend(movie):
    movie = movie.lower()
    try:
        movie_index = movies[movies['title'].str.lower() == movie].index[0]
    except IndexError:
        return ["Movie not found. Please try again."]
    
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended = [movies.iloc[i[0]]['title'] for i in movie_list]
    return recommended

# Streamlit UI
st.title("🎬 Movie Recommendation System")

movie_name = st.text_input("Enter a movie name:")

if st.button("Recommend"):
    if movie_name.strip() == "":
        st.warning("Please enter a movie name.")
    else:
        recommendations = recommend(movie_name)
        st.subheader("Recommended Movies:")
        for movie in recommendations:
            st.write(f"👉 {movie}")
