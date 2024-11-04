from train import Trainer
import pandas as pd 
import numpy as np
import tensorflow as tf 

class Validator():
    def __init__(self, train_model: Trainer, validate_set: pd.DataFrame): 
        self.fit(validate_set)
        self.user = train_model.user
        self.movie = train_model.movie
        self.bias_user = train_model.bias_user
        self.bias_movie = train_model.bias_movie
        self.result = self.evaluate()
        
    def fit(self, validate_set: pd.DataFrame):
        self.validate_set = [(np.int32(userId), np.int32(movieId), np.float64(rating)) for userId,movieId,rating in zip(validate_set['userId'], validate_set['movieId'], validate_set['rating'])]
        ratings = validate_set['rating'].to_numpy() 
        self.mean = np.mean(ratings)
        
    def predict(self, u, i)->float:
        return tf.tensordot(self.user[u], self.movie[i], axes=1) + self.bias_user[u] + self.bias_movie[i] + self.mean
    
    def evaluate(self)->float: 
        reals = tf.stack([tf.cast(row[2], tf.float64) for row in self.validate_set])
        preds = tf.stack([self.predict(row[0], row[1]) for row in self.validate_set])
        return tf.reduce_mean(tf.square(tf.subtract(reals ,preds)))