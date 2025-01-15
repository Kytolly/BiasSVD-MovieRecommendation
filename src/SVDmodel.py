import numpy as np
import tensorflow as tf  

class Model():
    def __init__(self, movies_num, users_num, k, lr):   
        self.movie = tf.Variable(tf.random.normal([movies_num, k], dtype=tf.float64))
        self.bias_movie = tf.Variable(tf.zeros([movies_num, 1], dtype=tf.float64))
        self.user = tf.Variable(tf.random.normal([users_num, k], dtype=tf.float64))
        self.bias_user = tf.Variable(tf.zeros([users_num, 1], dtype=tf.float64))
        self.policy = tf.keras.optimizers.SGD(lr)
        
    def fit(self, train_set):
        self.scale = len(train_set)
        self.usr_indices = tf.stack([tf.cast(row[0], tf.int32) for row in train_set])
        self.movie_indices= tf.stack([tf.cast(row[1], tf.int32) for row in train_set])  
        self.labels = tf.stack([tf.cast(row[2], tf.float64) for row in train_set])
        self.mean = tf.reduce_mean(self.labels)
    
    def cost(self, lambda_r):
        # embedding嵌入操作,提取子向量
        movie_objs = tf.nn.embedding_lookup(self.movie, self.movie_indices)
        user_objs = tf.nn.embedding_lookup(self.user, self.usr_indices)
        movie_bias_objs = tf.nn.embedding_lookup(self.bias_movie, self.movie_indices)
        user_bias_objs = tf.nn.embedding_lookup(self.bias_user, self.usr_indices) 
        
        # 计算正则项(norm2)
        user_norms = tf.square(tf.norm(user_objs, ord='euclidean', axis=1))
        movie_norms = tf.square(tf.norm(movie_objs, ord='euclidean', axis=1))
        user_bias_norm = tf.square(tf.norm(user_bias_objs, ord='euclidean', axis=1))
        movie_bias_norm = tf.square(tf.norm(movie_bias_objs, ord='euclidean', axis=1))
        regulation = tf.reduce_sum(user_norms + movie_norms + user_bias_norm + movie_bias_norm) 
        
        # 计算偏差项(SE)
        preds = tf.squeeze(tf.reduce_sum(tf.multiply(user_objs, movie_objs), axis=1, keepdims=True) + movie_bias_objs + user_bias_objs + self.mean)
        loss = tf.reduce_sum(tf.square(tf.subtract(self.labels, preds)))
        
        return loss + regulation * lambda_r    

    @tf.function
    def learn(self, lambda_r):
        with tf.GradientTape() as tape:
            cost = self.cost(lambda_r)
        gradients = tape.gradient(cost, [self.movie, self.user, self.bias_movie, self.bias_user])
        self.policy.apply_gradients(zip(gradients, [self.movie, self.user, self.bias_movie, self.bias_user]))
    
    def save(self, name):
        pass
    
if __name__ == "__main__": 
    train_set = [(1, 2, 2.5), (1, 3, 4.5), (2, 2, 5.0)]
    model = Model(5, 4, 3, 0.001)
    model.fit(train_set)
    for _ in range(10):
        model.learn(0.001)
        tf.print(model.movie)
        tf.print(model.user)
        tf.print(model.bias_movie)
        tf.print(model.bias_user)