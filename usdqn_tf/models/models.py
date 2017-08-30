import tensorflow as tf
from models.utils import weights, biases


class UsdqnModel(object):

    def __init__(self, dim_out, varscope=''):
        super(UsdqnModel, self).__init__()
        self.name = 'usdqn_%s' % (dim_out)
        self.dim_out = dim_out
        self.varscope = '%s_' % varscope if varscope != '' else ''

    def graph(self, state):

        with tf.variable_scope('conv1'):
            W_conv1 = weights([3, 3, 4, 16], '%sconv1' % self.varscope)
            b_conv1 = biases([16], '%sconv1' % self.varscope)
            out_conv1 = tf.nn.relu(
                            tf.nn.conv2d(state, W_conv1, 
                                strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

        with tf.variable_scope('conv2'):
            W_conv2 = weights([3, 3, 16, 32], '%sconv2' % self.varscope)
            b_conv2 = biases([32], '%sconv2' % self.varscope)
            out_conv2 = tf.nn.relu(
                            tf.nn.conv2d(out_conv1, W_conv2, 
                                strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

        with tf.variable_scope('conv3'):
            W_conv3 = weights([3, 3, 32, 64], '%sconv2' % self.varscope)
            b_conv3 = biases([64], '%sconv2' % self.varscope)
            out_conv3 = tf.nn.relu(
                            tf.nn.conv2d(out_conv2, W_conv3, 
                                strides=[1, 2, 2, 1], padding='SAME') + b_conv3)
            # Flatten conv 1 output
            o_shp = out_conv3.get_shape().as_list()
            conv3_dim = o_shp[1] * o_shp[2] * o_shp[3]
            out_conv3_flat = tf.reshape(out_conv3, [-1, conv3_dim])

        with tf.variable_scope('fully_connected'):
            W_fc = weights([conv3_dim, 1024], '%sfc' % self.varscope)
            b_fc = biases([1024], '%sfc' % self.varscope)
            out_fc = tf.nn.relu(tf.matmul(out_conv3_flat, W_fc) + b_fc)

        with tf.variable_scope('linear'):
            W_lin = weights([1024, self.dim_out], '%slin' % self.varscope)
            b_lin = biases([self.dim_out], '%slin' % self.varscope)
            out = tf.matmul(out_fc, W_lin) + b_lin
            #out = tf.maximum(tf.minimum(-3.05, out1), 3.05)
            # clip_min = tf.minimum(-3.05, out)
            # out = tf.group(clip_max, clip_min)
            # clip_b = out
            # clip = tf.group(clip_W, clip_b)

            # amax = tf.constant(3.05)
            # amin = tf.constant(-3.05)
            # def f1(): return out
            # def f21(): return tf.add(out, 6.10)
            # def f22(): return tf.add(out, -6.10)
            # def f2(): return tf.cond(tf.less(out, tf.constant(0)), f21, f22)
            # out = tf.cond(tf.less(tf.abs(out), amax), f1, f2)

        return out