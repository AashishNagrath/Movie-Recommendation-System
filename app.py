import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(
    page_title="Movie Recommender",
    layout="wide"
)
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    h1 {
        text-align: center;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        color: #bbbbbb;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        font-size: 0.9rem;
        color: #888888;
    }
    </style>
""", unsafe_allow_html=True)



@st.cache_data
def load_data():
    movies = pickle.load(open('movie_list.pkl', 'rb'))
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()
    similarity = cosine_similarity(vectors)
    return movies, similarity


movies, similarity = load_data()

movie_index_map = {title: idx for idx, title in enumerate(movies['title'])}


def fetch_poster(movie_id):
    API_KEY = st.secrets["TMDB_API_KEY"]
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
        else:
            return None
    except Exception as e:
        print("API Error:", e)
        return None


def recommend(movie):
    movie_index = movie_index_map[movie]
    distances = similarity[movie_index]
    
    movies_list = sorted(list(enumerate(distances)),
                         reverse=True,
                         key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    recommended_posters = []
    
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))
    
    return recommended_movies, recommended_posters


st.markdown("<h1>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Discover movies similar to your favorites instantly</div>", unsafe_allow_html=True)

selected_movie = st.selectbox(
    "Select a movie",
    movies['title'].values
)


if st.button("Recommend"):
    names, posters = recommend(selected_movie)

    st.markdown("### Recommended Movies")

    cols = st.columns(5)

    for i in range(5):
        with cols[i]:
            if posters[i]:
                st.image(posters[i], width=200)
            st.markdown(f"**{names[i]}**")
