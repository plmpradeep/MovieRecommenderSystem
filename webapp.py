from flask import Flask, render_template, request
app = Flask(__name__)

import numpy as np
import pandas as pd
import json as json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse.linalg import svds

# Loading movielens data

# Movies
movies = pd.read_csv('ml-latest-small/movies.csv')
# Ratings
ratings = pd.read_csv('ml-latest-small/ratings.csv')
ratings = ratings[['userId', 'movieId', 'rating']]
# taking a 1MM sample because it can take too long to pivot data later on
ratings = ratings.head(100000)
# convert data types before merging
movies.movieId = pd.to_numeric(movies.movieId, errors='coerce')
ratings.movieId = pd.to_numeric(ratings.movieId, errors='coerce')

#svd
n_users = ratings.userId.unique().shape[0]
n_movies = ratings.movieId.unique().shape[0]
Ratings = ratings.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
R = Ratings.as_matrix()
user_ratings_mean = np.mean(R, axis = 1)
Ratings_demeaned = R - user_ratings_mean.reshape(-1, 1)
U, sigma, Vt = svds(Ratings_demeaned, k = 50)
sigma = np.diag(sigma)
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds = pd.DataFrame(all_user_predicted_ratings, columns = Ratings.columns)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/svd")
def svd():
    return render_template('svd.html')


@app.route("/movies")
def getmovies():
    movieList = list(movies['title'].values.flatten())
    return json.dumps(movieList)


@app.route("/tfidf", methods=['POST'])
def getTFIDMovies():
    user_inp = request.json['movie']
    tf = TfidfVectorizer(analyzer='word', ngram_range=(
        1, 2), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(movies['genres'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    titles = movies['title']
    indices = pd.Series(movies.index, index=movies['title'])
    idx = indices[user_inp]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    movieList = []
    movieList = list(titles.iloc[movie_indices])
    return json.dumps(movieList)

# A simple way to compute Pearson Correlation


def pearsonR(s1, s2):
    s1_c = s1 - s1.mean()
    s2_c = s2 - s2.mean()
    return np.sum(s1_c * s2_c) / np.sqrt(np.sum(s1_c**2) * np.sum(s2_c**2))


@app.route("/pearson", methods=['POST'])
def getPearsonMovies():
    user_inp = request.json['movie']
    # create a single dataset merging the previous 2
    data = pd.merge(ratings, movies, on='movieId', how='inner')
    matrix = data.pivot_table(index='userId', columns='title', values='rating')
    similarMovies = []
    for title in matrix.columns:
        if title == user_inp:
            continue
        cor = pearsonR(matrix[user_inp], matrix[title])
        if np.isnan(cor):
            continue
        else:
            similarMovies.append((title, cor))

    similarMovies.sort(key=lambda tup: tup[1], reverse=True)
    tmpList = [item[0] for item in similarMovies]
    movieList = list(tmpList[:10])
    return json.dumps(movieList)

#SVD Algorithm
@app.route("/svdByUserID", methods=['POST'])
def getSVDMovies():
    userID = request.json['userID']
    userID=int(userID)
    user_row_number = userID - 1 # User ID starts at 1, not 0
    sorted_user_predictions = preds.iloc[user_row_number].sort_values(ascending=False) # User ID starts at 1

    # Get the user's data and merge in the movie information.
    user_data = ratings[ratings.userId == (userID)]
    user_full = (user_data.merge(movies, how = 'left', left_on = 'movieId', right_on = 'movieId').sort_values(['rating'], ascending=False))

    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies[~movies['movieId'].isin(user_full['movieId'])].merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',left_on = 'movieId',right_on = 'movieId').rename(columns = {user_row_number: 'Predictions'}).sort_values('Predictions', ascending = False).iloc[:10, :-1])
    

    return json.dumps(movieList)


if __name__ == "__main__":
    app.run(debug=True)
