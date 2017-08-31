import tensorflow as tf


def weights(shape, name):
    return tf.get_variable('W_%s' % name, shape, 
				initializer=tf.contrib.layers.xavier_initializer())

def biases(shape, name):
    return tf.get_variable('b_%s' % name, shape, 
    			initializer=tf.contrib.layers.xavier_initializer())