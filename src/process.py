import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  
import tensorflow as tf 

class Processer():
    def __init__(self, movies_path: str, ratings_path: str, underlying_features_K: int):
        self.loadMovies(movies_path, underlying_features_K)
        self.loadUsers(ratings_path, underlying_features_K)
        ratings = self.loadRatings(ratings_path, underlying_features_K)
        self.split(ratings)
    
    def loadMovies(self, movies_path: str, underlying_features_K: int):
        movies = pd.read_csv(movies_path, sep=',', encoding='utf-8')
        movie_col = movies.loc[:, 'movieId'].unique() 
        movies_num = movie_col.size
        sorted(movie_col)
        self.movie_dict = {movie_id: idx for idx,movie_id in enumerate(movie_col)} 
        self.movie = tf.Variable(tf.random.normal([movies_num, underlying_features_K], dtype=tf.float64))
        self.bias_movie = tf.Variable(tf.random.normal([movies_num, 1], dtype=tf.float64))

    def loadUsers(self, ratings_path: str, underlying_features_K: int):
        ratings = pd.read_csv(ratings_path, sep=',', encoding='utf-8')
        user_col = ratings.loc[:, 'userId'].unique() 
        users_num = user_col.size 
        sorted(user_col)
        self.user_dict = {user_id: idx for idx,user_id in enumerate(user_col)} 
        self.user = tf.Variable(tf.random.normal([users_num, underlying_features_K], dtype=tf.float64))
        self.bias_user = tf.Variable(tf.random.normal([users_num, 1], dtype=tf.float64))
        
    def loadRatings(self, ratings_path: str, underlying_features_K: int):  
        ratings = pd.read_csv(ratings_path, sep=',', encoding='utf-8') 
        movie_col = ratings.loc[:, 'movieId'].to_numpy()
        new_movie_col = [self.movie_dict[movie_id] for movie_id in movie_col]
        ratings['movieId'] = new_movie_col
        user_col = ratings.loc[:, 'userId'].to_numpy()
        new_user_col = [self.user_dict[user_id] for user_id in user_col]
        ratings['userId'] = new_user_col
        return ratings
        
    def split(self, ratings):
        self.dataset_slices = [] 
        for i in range(10, 1, -1):
            ratings = train_test_split(ratings, test_size=1/i, random_state=0, shuffle=True) 
            self.dataset_slices.append(ratings[1])
            ratings = ratings[0]
            if i == 2:
                self.dataset_slices.append(ratings)
                
if __name__ == '__main__': 
    movies_path = 'data/small/movie.csv'
    ratings_path = 'data/small/rating.csv'
    p = Processer(movies_path, ratings_path, 3)
    print(p.movie_dict)
    print(p.movie)
    print(p.bias_movie)
    print(p.user_dict)
    print(p.user)
    print(p.bias_user)
    print(p.dataset_slices) 