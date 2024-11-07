import numpy as np
import tensorflow as tf  

class Model():
    def __init__(self, movies_num, users_num, k, lr):   
        self.movie = tf.Variable(tf.random.normal([movies_num, k], dtype=tf.float64))
        self.bias_movie = tf.Variable(tf.random.normal([movies_num, 1], dtype=tf.float64))
        self.user = tf.Variable(tf.random.normal([users_num, k], dtype=tf.float64))
        self.bias_user = tf.Variable(tf.random.normal([users_num, 1], dtype=tf.float64))
        self.policy = tf.keras.optimizers.SGD(lr)
        
    def fit(self, train_set):
        self.mean = np.mean([r[2] for r in train_set]) 
        
    def predict(self, u, i):
        part1 = tf.tensordot(self.user[u], self.movie[i], axes=1)
        part2 = self.bias_user[u] + self.bias_movie[i]
        part3 = tf.cast(self.mean, tf.float64)
        res =  tf.squeeze(part1 + part2 + part3)
        return res
    
    def normSum(self, u, i):
        return sum(x**2 for x in [tf.norm(self.user[u]), tf.norm(self.movie[i]), tf.norm(self.bias_user[u]), tf.norm(self.bias_movie[i])])
                           
    def regulation(self, train_set):  
        return tf.reduce_sum(tf.stack([self.normSum(row[0], row[1]) for row in train_set]))
    
    def loss(self, train_set): 
        reals = tf.stack([tf.cast(row[2], tf.float64) for row in train_set])
        preds = tf.stack([self.predict(row[0], row[1]) for row in train_set])
        return tf.reduce_sum(tf.square(tf.subtract(reals ,preds)))
    
    def cost(self, train_set, lambda_r):
        return self.loss(train_set) + self.regulation(train_set) * lambda_r    

    @tf.function
    def learn(self, train_set, lambda_r):
        with tf.GradientTape() as tape:
            cost = self.cost(train_set, lambda_r)
        gradients = tape.gradient(cost, [self.movie, self.user, self.bias_movie, self.bias_user])
        self.policy.apply_gradients(zip(gradients, [self.movie, self.user, self.bias_movie, self.bias_user]))
    
    def save(self, name):
        pass