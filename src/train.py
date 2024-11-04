from parameter import Parameters 
import pandas as pd 
import numpy as np
import tensorflow as tf 

class Trainer(Parameters):
    def __init__(self, config_path: str, movies_num, users_num): 
        self.compile(config_path)
        self.movie = tf.Variable(tf.random.normal([movies_num, self.underlying_features_K], dtype=tf.float64))
        self.bias_movie = tf.Variable(tf.random.normal([movies_num, 1], dtype=tf.float64))
        self.user = tf.Variable(tf.random.normal([users_num, self.underlying_features_K], dtype=tf.float64))
        self.bias_user = tf.Variable(tf.random.normal([users_num, 1], dtype=tf.float64))
        
    def compile(self, config_path: str) :
        super().__init__(config_path)
        
    def fit(self, train_set: pd.DataFrame): 
        # train_set['userId'] = train_set['userId'].astype(int)
        # train_set['movieId'] = train_set['movieId'].astype(int) 
        self.train_set = [(np.int32(userId), np.int32(movieId), np.float64(rating)) for userId,movieId,rating in zip(train_set['userId'], train_set['movieId'], train_set['rating'])]
        ratings = train_set['rating'].to_numpy()
        self.mean = np.mean(ratings)
    
    def normSum(self, u, i)->float:
        return sum(x**2 for x in [tf.norm(self.user[u]), tf.norm(self.movie[i]), tf.norm(self.bias_user[u]), tf.norm(self.bias_movie[i])])
    
    def predict(self, u, i)->tf.Tensor:
        return tf.tensordot(self.user[u], self.movie[i], axes=1) + self.bias_user[u] + self.bias_movie[i] + self.mean
        
    def regulation(self): 
        reg = tf.stack([self.normSum(row[0], row[1]) for row in self.train_set])
        return tf.reduce_sum(reg)
    
    def loss(self): 
        reals = tf.stack([tf.cast(row[2], tf.float64) for row in self.train_set])
        preds = tf.stack([self.predict(row[0], row[1]) for row in self.train_set])
        return tf.reduce_sum(tf.square(tf.subtract(reals ,preds)))
    
    def loss_test(self):
        errors = []
        for row in self.train_set:
            u = tf.cast(row[0], tf.int32)
            i = tf.cast(row[1], tf.int32)
            pred = self.predict(u, i)
            real = row[2]
            errors.append(real - pred)
            print(f"Predicted: {pred[0]}, Real: {real}")
        errors = tf.stack(errors)
        return tf.reduce_sum(tf.square(errors))
    
    def cost(self):
        return self.loss() + self.regulation() *self.lambda_r    

    @tf.function
    def train_step(self, optimizer):
        with tf.GradientTape() as tape:   
            cost = self.cost()
        gradients = tape.gradient(cost, [self.movie, self.user, self.bias_movie, self.bias_user])
        optimizer.apply_gradients(zip(gradients, [self.movie, self.user, self.bias_movie, self.bias_user]))

    def train(self, train_set: pd.DataFrame):
        self.fit(train_set) 
        optimizer = tf.keras.optimizers.SGD(self.learning_rate)
        for _ in range(self.steps):
            self.train_step(optimizer)     
               
if __name__ == "__main__":
    movies_path = 'data/small/movie.csv'
    ratings_path = 'data/small/rating.csv'
    from process import Processer
    p = Processer(movies_path, ratings_path, 3)
    t = Trainer('config/config.yaml', p.movie, p.user, p.bias_movie, p.bias_user) 
    train_set = pd.concat(p.dataset_slices[:]) 
    t.fit(train_set) 
    print(f'mean: {t.mean}')
    t.train(train_set)
    print(t.movie)
    print(t.user)
    print(t.bias_movie)
    print(t.bias_user)
    
    # with tf.GradientTape() as tape:   
    #     reals = tf.stack([tf.cast(row[2], tf.float64) for row in t.train_set])
    #     preds = tf.stack([t.predict(row[0], row[1]) for row in t.train_set])
    #     print(f'reals: {tf.get_static_value(reals)}')
    #     print(f'preds: {tf.get_static_value(preds)}')
    #     cost = tf.reduce_sum(tf.square(tf.subtract(reals ,preds)))
    #     print(f'cost: {tf.get_static_value(cost)}')
    # gradients = tape.gradient(cost, [t.movie, t.user, t.bias_movie, t.bias_user])
    # for gra in gradients:
    #     print(f'gradients: {tf.get_static_value(gra)}')
    # optimizer.apply_gradients(zip(gradients, [t.movie, t.user, t.bias_movie, t.bias_user]))