from SVDmodel import Model
import tensorflow as tf 

class Tester():
    def __init__(self, model: Model, test_set): 
        model.fit(test_set)
        self.result = self.evaluate(model)

    def evaluate(self, model: Model)->float: 
        movie_objs = tf.nn.embedding_lookup(model.movie, model.movie_indices)
        user_objs = tf.nn.embedding_lookup(model.user, model.usr_indices)
        movie_bias_objs = tf.nn.embedding_lookup(model.bias_movie, model.movie_indices)
        user_bias_objs = tf.nn.embedding_lookup(model.bias_user, model.usr_indices)
        
        preds = tf.squeeze(tf.reduce_sum(tf.multiply(user_objs, movie_objs), axis=1, keepdims=True) + movie_bias_objs + user_bias_objs + model.mean)
        mse = tf.reduce_mean(tf.square(tf.subtract(model.labels, preds))) 
        return mse