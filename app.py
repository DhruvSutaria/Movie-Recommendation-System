import streamlit as st
import pandas as pd
from thefuzz import process
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
import requests
def fetch_poster(movie_id):
    data = requests.get("https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id))
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path


def recommend_movie(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    recommended_movie = []
    recommended_movie_poster = []
    for i in distances[1:6]:
        movies_id = movies.iloc[i[0]].movie_id
        recommended_movie.append(movies.iloc[i[0]].title)
        recommended_movie_poster.append(fetch_poster(movies_id))
    return recommended_movie, recommended_movie_poster


movies = pd.read_pickle("./dummy.pkl")
movie_list = movies['title'].values

similarity = pickle.load(open("./similarity.pkl", "rb"))

def movie_finder(title):
    all_titles = movies_df['title'].tolist()
    closest_match = process.extractOne(title,all_titles)
    return closest_match[0]

def movie_finders(title):
    all_titles = movies['title'].tolist()
    closest_match = process.extractOne(title,all_titles)
    return closest_match[0]

def find_similar_movies(title, X, movie_mapper, movie_inv_mapper, k, metric='cosine'):

    X = X.T
    neighbour_ids = []
    movie_id = movies_df[movies_df['title'] == title].iloc[0, 0]
    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    if isinstance(movie_vec, (np.ndarray)):
        movie_vec = movie_vec.reshape(1, -1)
    # use k+1 since kNN output includes the movieId of interest
    kNN = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric=metric)
    kNN.fit(X)
    neighbour = kNN.kneighbors(movie_vec, return_distance=False)
    for i in range(0, k):
        n = neighbour.item(i)
        neighbour_ids.append(movie_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids

movies_df = pd.read_pickle("./movie_df.pkl")
csr_mat = pickle.load(open("./csr_mat.pkl", "rb"))
movie_mapper = pickle.load(open("./movie_mapper.pkl", "rb"))
movie_inv_mapper = pickle.load(open("./movie_inv_mapper.pkl", "rb"))
movie_titles = dict(zip(movies_df['movieId'], movies_df['title']))

st.title('Movie Recommendation (System Content Based)')
movie_title_cb = st.text_input("Enter a movie's title")
movie_real_title_cb = movie_finders(movie_title_cb)

if st.button('Recommend one'):
    rec_name,rec_posters = recommend_movie(movie_real_title_cb)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(rec_name[0])
        st.image(rec_posters[0])
    with col2:
        st.text(rec_name[1])
        st.image(rec_posters[1])

    with col3:
        st.text(rec_name[2])
        st.image(rec_posters[2])
    with col4:
        st.text(rec_name[3])
        st.image(rec_posters[3])
    with col5:
        st.text(rec_name[4])
        st.image(rec_posters[4])

st.title('Movie Recommendation System Collaborative Filtering')
st.write("Enter a movie title")
movie_title = st.text_input("Enter a movie title")

if st.button('Recommend one to me'):
    movie_real_title = movie_finder(movie_title)
    recommend_name = find_similar_movies(movie_real_title, csr_mat, movie_mapper, movie_inv_mapper, metric='cosine', k=10)
    for i in recommend_name:
        st.write(movie_titles[i])





