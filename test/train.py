from parameter import Parameters 
import pandas as pd 
import numpy as np
import tensorflow as tf 
from process import Processer 

class Trainer(Parameters):
    def __init__(self, config_path: str, movies_num, users_num): 
        self.compile(config_path)
        self.optimizer = tf.keras.optimizers.SGD(self.learning_rate)
        self.movie = tf.Variable(tf.random.normal([movies_num, self.underlying_features_K], dtype=tf.float64))
        self.bias_movie = tf.Variable(tf.random.normal([movies_num, 1], dtype=tf.float64))
        self.user = tf.Variable(tf.random.normal([users_num, self.underlying_features_K], dtype=tf.float64))
        self.bias_user = tf.Variable(tf.random.normal([users_num, 1], dtype=tf.float64))
        
    def compile(self, config_path: str) :
        super().__init__(config_path)
        print(f'the paras is {str(super())}')
        
    def fit(self, train_set: pd.DataFrame): 
        # train_set['userId'] = train_set['userId'].astype(int)
        # train_set['movieId'] = train_set['movieId'].astype(int) 
        self.train_set = [(int(userId), int(movieId), np.float64(rating)) for userId,movieId,rating in zip(train_set['userId'], train_set['movieId'], train_set['rating'])]
        print(f'fitting the train_set(has {len(self.train_set)} records)...')
        print(f'the train_set is {self.train_set}')
        
        ratings = train_set['rating'].to_numpy()
        self.mean = np.mean(ratings)
        print(f'the batch of train_set average value is {self.mean}')
    
    def normSum(self, u, i)->float:
        return sum(x**2 for x in [tf.norm(self.user[u]), tf.norm(self.movie[i]), tf.norm(self.bias_user[u]), tf.norm(self.bias_movie[i])])
    
    def predict(self, u, i)->tf.Tensor:  
        return tf.squeeze(tf.tensordot(self.user[u], self.movie[i], axes=1) + self.bias_user[u] + self.bias_movie[i] + tf.cast(self.mean, tf.float64))
        
    def regulation(self):  
        return tf.reduce_sum(tf.stack([self.normSum(row[0], row[1]) for row in self.train_set]))
    
    def loss(self): 
        reals = tf.stack([tf.cast(row[2], tf.float64) for row in self.train_set])
        preds = tf.stack([self.predict(row[0], row[1]) for row in self.train_set])
        return tf.reduce_sum(tf.square(tf.subtract(reals ,preds)))
    
    def cost(self):
        return self.loss() + self.regulation() *self.lambda_r    

    @tf.function
    def train_step(self, optimizer):
        print('----------------------------------------------------------------')
        with tf.GradientTape() as tape:   
            reals = tf.stack([tf.cast(row[2], tf.float64) for row in self.train_set])
            preds = tf.stack([self.predict(row[0], row[1]) for row in self.train_set]) 
            cost = tf.reduce_sum(tf.square(tf.subtract(reals ,preds))) 
            
        gradients = tape.gradient(cost, [self.movie, self.user, self.bias_movie, self.bias_user])
        optimizer.apply_gradients(zip(gradients, [self.movie, self.user, self.bias_movie, self.bias_user]))
        print('----------------------------------------------------------------')
        
    def train(self, train_set: pd.DataFrame):
        self.fit(train_set)  
        for _ in range(self.steps):
            self.train_step(self.optimizer)   
            
if __name__ == '__main__':
    process = Processer('data/small/movies.csv', 'data/small/ratings.csv')
    trainer = Trainer('config/config.yaml', process.movies_num, process.users_num)  
    Rating_train = pd.concat(process.dataset_slices[1:1] +process.dataset_slices[2:])
    trainer.train(Rating_train)
        