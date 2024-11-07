from SVDmodel import Model
import tensorflow as tf 

class Tester():
    def __init__(self, model: Model, test_set): 
        model.fit(test_set)
        self.result = self.evaluate(model, test_set)

    def evaluate(self, model: Model, test_set)->float: 
        reals = tf.stack([tf.cast(row[2], tf.float64) for row in test_set])
        preds = tf.stack([model.predict(row[0], row[1]) for row in test_set])
        return tf.reduce_mean(tf.square(tf.subtract(reals ,preds)))