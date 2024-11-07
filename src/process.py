import pandas as pd
import numpy as np
from itertools import chain
from sklearn.model_selection import train_test_split

class Processer():
    def __init__(self, movies_path, ratings_path: str):
        self.movies_num = 0
        self.users_num = 0
        self.movie_dict = {}
        self.user_dict = {}
        self.dataset_slices = [] 
        
        self.loadMovies(movies_path)
        self.loadUsers(ratings_path)
        ratings = self.loadRatings(ratings_path)
        self.split(ratings)
    
    def loadMovies(self, movies_path: str):
        movies = pd.read_csv(movies_path, sep=',', encoding='utf-8')
        movie_col = movies.loc[:, 'movieId'].unique() 
        self.movies_num = movie_col.size
        print(f'the movies number is {self.movies_num}')
        
        sorted(movie_col)
        self.movie_dict = {movie_id: idx for idx,movie_id in enumerate(movie_col)} 
        print(f'the movie_dict is {self.movie_dict}')

    def loadUsers(self, ratings_path: str):
        ratings = pd.read_csv(ratings_path, sep=',', encoding='utf-8')
        user_col = ratings.loc[:, 'userId'].unique() 
        self.users_num = user_col.size 
        print(f'the users number is {self.users_num}')
        
        sorted(user_col)
        self.user_dict = {user_id: idx for idx,user_id in enumerate(user_col)} 
        print(f'the user_dict is {self.user_dict}')
        
    def loadRatings(self, ratings_path: str):  
        # 优化
        ratings = pd.read_csv(ratings_path, sep=',', encoding='utf-8') 
        
        movie_col = ratings.loc[:, 'movieId'].to_numpy()
        new_movie_col = [self.movie_dict[movie_id] for movie_id in movie_col]
        ratings['movieId'] = new_movie_col
        
        user_col = ratings.loc[:, 'userId'].to_numpy()
        new_user_col = [self.user_dict[user_id] for user_id in user_col]
        ratings['userId'] = new_user_col
        
        ratings = [(int(userId), int(movieId), np.float64(rating)) for userId,movieId,rating 
                   in zip(ratings['userId'], ratings['movieId'], ratings['rating'])]
        
        print(f'the rating is loaded as {ratings}')
        return ratings
        
    def split(self, ratings):
        for i in range(10, 1, -1):
            print(f'the split range in {i}')
            
            ratings = train_test_split(ratings, test_size=1/i, random_state=0, shuffle=True) 
            self.dataset_slices.append(ratings[1])
            print(f'append {len(ratings[1])} records in slices')
            
            ratings = ratings[0]
            if i == 2:
                self.dataset_slices.append(ratings)
                
        print(f'the slices length is {len(self.dataset_slices)}')
    
    def cross(self, i):
        # train-validate-test
        return (
            self.dataset_slices[1:i] + self.dataset_slices[i+1:],
            self.dataset_slices[i],
            self.dataset_slices[0]
        )
    
    def getTrainSet(self, i): 
        return list(chain.from_iterable(self.dataset_slices[j] for j in range(1, 10) if j != i))
    
    def getValidateSet(self, i):
        return self.dataset_slices[i]
    
    def getTestSet(self):
        return  self.dataset_slices[0]