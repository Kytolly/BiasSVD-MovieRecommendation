import tensorflow as tf

x = tf.constant([[1,2,3],
               [4,5,6],
               [7,8,9],
               [1,2,3],
               [4,5,6],
               [7,8,9]])
y = tf.constant([[1,2,3],
               [4,5,6],
               [7,8,9],
               [1,2,3],
               [4,5,6],
               [7,8,9]])
tmp = tf.multiply(x, y) 
tf.print(tmp)
res = tf.reduce_sum(tmp, axis=1, keepdims=True)
tf.print(res)
preds = tf.squeeze(res)
tf.print(preds)
labels = tf.stack([1,2,3,4,5,6])
tf.print(labels)
loss = tf.reduce_sum(tf.square(tf.subtract(labels, preds)))
tf.print(loss)

