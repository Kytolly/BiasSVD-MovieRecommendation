import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer

class MovieLayer(Layer):
    def __init__(self, movies_num, k, **kwargs):
        super(MovieLayer, self).__init__(**kwargs)
        self.movie = tf.Variable(tf.random.normal([movies_num, k], dtype=tf.float64), trainable=True)
        self.bias_movie = tf.Variable(tf.random.normal([movies_num, 1], dtype=tf.float64), trainable=True)

    def call(self, inputs):
        movie_id = tf.cast(inputs, tf.int32)
        return self.movie[movie_id], self.bias_movie[movie_id]

class UserLayer(Layer):
    def __init__(self, users_num, k, **kwargs):
        super(UserLayer, self).__init__(**kwargs)
        self.user = tf.Variable(tf.random.normal([users_num, k], dtype=tf.float64), trainable=True)
        self.bias_user = tf.Variable(tf.random.normal([users_num, 1], dtype=tf.float64), trainable=True)

    def call(self, inputs):
        user_id = tf.cast(inputs, tf.int32)
        return self.user[user_id], self.bias_user[user_id]

class Model(Model):
    def __init__(self, movies_num, users_num, k, lr, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.movie_layer = MovieLayer(movies_num, k)
        self.user_layer = UserLayer(users_num, k)
        self.mean = tf.Variable(0.0, dtype=tf.float64, trainable=False)
        self.policy = tf.keras.optimizers.SGD(lr)
        self.lambda_r = tf.Variable(0.0, dtype=tf.float64, trainable=False)

    def fit(self, train_set):
        self.mean.assign(np.mean([r[2] for r in train_set]))

    def predict(self, u, i):
        user, bias_user = self.user_layer(u)
        movie, bias_movie = self.movie_layer(i)
        part1 = tf.tensordot(user, movie, axes=1)
        part2 = bias_user + bias_movie
        part3 = tf.cast(self.mean, tf.float64)
        res = tf.squeeze(part1 + part2 + part3)
        return res

    def normSum(self, u, i):
        user, _ = self.user_layer(u)
        movie, _ = self.movie_layer(i)
        return sum(x**2 for x in [tf.norm(user), tf.norm(movie), tf.norm(self.user_layer.user[u]), tf.norm(self.movie_layer.movie[i])])

    def regulation(self, train_set):
        return tf.reduce_sum(tf.stack([self.normSum(row[0], row[1]) for row in train_set]))

    def loss(self, train_set):
        reals = tf.stack([tf.cast(row[2], tf.float64) for row in train_set])
        preds = tf.stack([self.predict(row[0], row[1]) for row in train_set])
        return tf.reduce_sum(tf.square(tf.subtract(reals, preds)))

    def cost(self, train_set):
        return self.loss(train_set) + self.regulation(train_set) * self.lambda_r

    @tf.function
    def learn(self, train_set):
        with tf.GradientTape() as tape:
            cost = self.cost(train_set)
        gradients = tape.gradient(cost, self.trainable_variables)
        self.policy.apply_gradients(zip(gradients, self.trainable_variables))

    def save(self, name):
        self.save_weights(name, save_format='h5')

    def load(self, name):
        self.load_weights(name)