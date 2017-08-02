import tensorflow as tf
import numpy as np

import warnings

from skimage.color import rgb2gray
from skimage.transform import resize
from scipy import ndimage
from sklearn.utils import shuffle

import matplotlib.pyplot as plt

FRAME_WIDTH = 84
FRAME_HEIGHT = 84
FRAME_BUFFER_SIZE = 1
DIM_OUT = 1


def do_obs_processing(frame, width, height):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return resize(rgb2gray(frame), (width, height))


def weights(shape, name):
    return tf.get_variable('W_%s' % name, shape, 
                initializer=tf.truncated_normal_initializer(
                                mean=0.0, stddev=0.01, dtype=tf.float32))

def biases(shape, name):
    return tf.get_variable('b_%s' % name, shape, 
                initializer=tf.constant_initializer(0.1))


def get_graph_old(state):
    # with tf.variable_scope('conv1'):
    #     W_conv1 = weights([6, 6, FRAME_BUFFER_SIZE, 16], 'conv1')
    #     b_conv1 = biases([16], 'conv1')
    #     out_conv1 = tf.nn.relu(
    #                     tf.nn.conv2d(state, W_conv1, 
    #                         strides=[1, 2, 2, 1], padding='SAME') + b_conv1)

    with tf.variable_scope('conv2'):
        W_conv2 = weights([3, 3, FRAME_BUFFER_SIZE, 4], 'conv2')
        b_conv2 = biases([4], 'conv2')
        out_conv2 = tf.nn.relu(
                        tf.nn.conv2d(state, W_conv2, 
                            strides=[1, 2, 2, 1], padding='SAME') + b_conv2)

    with tf.variable_scope('conv3'):
        W_conv3 = weights([3, 3, 4, 8], 'conv3')
        b_conv3 = biases([8], 'conv3')
        out_conv3 = tf.nn.relu(
                        tf.nn.conv2d(out_conv2, W_conv3, 
                            strides=[1, 2, 2, 1], padding='SAME') + b_conv3)

    with tf.variable_scope('conv4'):
        W_conv4 = weights([3, 3, 8, 16], 'conv4')
        b_conv4 = biases([16], 'conv4')
        out_conv4 = tf.nn.relu(
                        tf.nn.conv2d(out_conv3, W_conv4, 
                            strides=[1, 2, 2, 1], padding='SAME') + b_conv4)

    with tf.variable_scope('conv5'):
        W_conv5 = weights([3, 3, 16, 32], 'conv5')
        b_conv5 = biases([32], 'conv5')
        out_conv5 = tf.nn.relu(
                        tf.nn.conv2d(out_conv4, W_conv5, 
                            strides=[1, 2, 2, 1], padding='SAME') + b_conv5)

    # with tf.variable_scope('conv2'):
    #     W_conv2 = weights([4, 4, 16, 32], 'conv2')
    #     b_conv2 = biases([32], 'conv2')
    #     out_conv2 = tf.nn.relu(
    #                     tf.nn.conv2d(out_conv1, W_conv2, 
    #                         strides=[1, 2, 2, 1], padding='SAME') + b_conv2)
        # Flatten conv 1 output
        o_shp = out_conv5.get_shape().as_list()
        print("Conv shape:", out_conv5.get_shape())
        conv2_dim = o_shp[1] * o_shp[2] * o_shp[3]
        out_conv2_flat = tf.reshape(out_conv5, [-1, conv2_dim])

    with tf.variable_scope('fully_connected'):
        W_fc = weights([conv2_dim, 1024], 'fc')
        b_fc = biases([1024], 'fc')
        out_fc = tf.nn.relu(tf.matmul(out_conv2_flat, W_fc) + b_fc)
        #out_fc = tf.nn.relu(tf.matmul(out_conv2_flat, W_fc) + b_fc)

    with tf.variable_scope('fully_connected2'):
        W_fc2 = weights([1024, 2048], 'fc2')
        b_fc2 = biases([2048], 'fc2')
        out_fc2 = tf.nn.relu(tf.matmul(out_fc, W_fc2) + b_fc2)

    with tf.variable_scope('linear'):
        W_lin = weights([2048, DIM_OUT], 'lin')
        b_lin = biases([DIM_OUT], 'lin')
        out = tf.matmul(out_fc2, W_lin) + b_lin

    return out_fc


def get_graph(state):

        with tf.variable_scope('conv1'):
            W_conv1 = weights([6, 6, FRAME_BUFFER_SIZE, 16], 'conv1')
            b_conv1 = biases([16], 'conv1')
            out_conv1 = tf.nn.relu(
                            tf.nn.conv2d(state, W_conv1, 
                                strides=[1, 2, 2, 1], padding='SAME') + b_conv1)

        with tf.variable_scope('conv2'):
            W_conv2 = weights([4, 4, 16, 32], 'conv2')
            b_conv2 = biases([32], 'conv2')
            out_conv2 = tf.nn.relu(
                            tf.nn.conv2d(out_conv1, W_conv2, 
                                strides=[1, 2, 2, 1], padding='SAME') + b_conv2)
            # Flatten conv 1 output
            o_shp = out_conv2.get_shape().as_list()
            conv2_dim = o_shp[1] * o_shp[2] * o_shp[3]
            out_conv2_flat = tf.reshape(out_conv2, [-1, conv2_dim])

        with tf.variable_scope('fully_connected'):
            W_fc = weights([conv2_dim, 256], 'fc')
            b_fc = biases([256], 'fc')
            out_fc = tf.nn.relu(tf.matmul(out_conv2_flat, W_fc) + b_fc)

        with tf.variable_scope('linear'):
            W_lin = weights([256, DIM_OUT], 'lin')
            b_lin = biases([DIM_OUT], 'lin')
            out = tf.matmul(out_fc, W_lin) + b_lin

        return out

def run():

    tf.reset_default_graph()


    # Create placeholders
    states_pl = tf.placeholder(tf.float32, 
        shape=(None, FRAME_WIDTH, FRAME_HEIGHT, FRAME_BUFFER_SIZE), name='states')
    actions_pl= tf.placeholder(tf.float32, shape=(None), name='actions')

    q_output = get_graph(states_pl)

    # # Compute Q from current q_output and one hot actions
    # Q = tf.reduce_sum(
    #         tf.multiply(q_output, 
    #             tf.one_hot(actions_pl, env.action_space.n, dtype=tf.float32)
    #         ), axis=1)

    # # Loss operation 
    loss_op = tf.reduce_mean(tf.square(actions_pl - q_output) / 2)
    
    # # Optimizer Op
    # #optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

    # # Training Op
    train_op = optimizer.minimize(loss_op)

    # # Prediction Op
    if DIM_OUT == 1:
        prediction = q_output
    else:
        prediction = tf.argmax(q_output, 1)

    # Model Saver
    saver = tf.train.Saver()

    # init all variables
    init_op = tf.global_variables_initializer()

    # Limit memory usage for multiple training at same time
    config = tf.ConfigProto(allow_soft_placement=True)
    #config.gpu_options.per_process_gpu_memory_fraction = 0.33

    # Start Session
    with tf.Session(config=config) as sess:
        sess.run(init_op)

        print('GRAPH is loaded')

        images = np.load('../data/1dof/usdqn-images-training.npy')
        labels = np.load('../data/1dof/usdqn-labels-training.npy')

        print(images.shape)

        images_n = list()

        # import matplotlib.cm as cm
        for k in [0, 200, 300, 400, 500, 600]:
        #     plt.imsave("im-%s.jpeg" % k, images[k,:,:], cmap=cm.gray)
            img = ndimage.imread("im-%s.jpeg" % k) / 255
            images_n.append(do_obs_processing(img[10:,5:-5], 84, 84))

        for img in images:
            img = do_obs_processing(img[10:,5:-5] / 255, 84, 84)

        # images_n = np.stack(images_n)
        # print('images:', images_n.shape)

        # f, axarr = plt.subplots(3, 2)
        # axarr[0, 0].imshow(images_n[0,:,:])
        # axarr[0, 1].imshow(images_n[1,:,:])
        # axarr[1, 0].imshow(images_n[2,:,:])
        # axarr[1, 1].imshow(images_n[3,:,:])
        # axarr[2, 0].imshow(images_n[4,:,:])
        # axarr[2, 1].imshow(images_n[5,:,:])
        # plt.pause(1)
        batch_size = 32

        for epoch in range(100):
            print("Epoch %s/30" % epoch)
            e_images, e_labels = shuffle(images, labels)

            for batch in range(int(len(images) / batch_size)):
                c_images = e_images[batch*batch_size:(batch+1)*batch_size, :, :].reshape([-1, 84, 84, 1])
                c_labels = e_labels[batch*batch_size:(batch+1)*batch_size]
                loss, _ = sess.run([loss_op, train_op], 
                                feed_dict={
                                    states_pl: c_images,
                                    actions_pl: c_labels,
                                })

                #print(loss)


        test_images = np.load('../data/1dof/usdqn-images-testing.npy')
        test_labels = np.load('../data/1dof/usdqn-labels-testing.npy')

        for img in test_images:
            img = do_obs_processing(img[10:,5:-5] / 255, 84, 84)

        error = 0
        for batch in range(int(len(test_images) / batch_size)):
            c_images = test_images[batch*batch_size:(batch+1)*batch_size, :, :].reshape([-1, 84, 84, 1])
            c_labels = test_labels[batch*batch_size:(batch+1)*batch_size]
            error += sess.run(loss_op, 
                                feed_dict={
                                    states_pl: c_images,
                                    actions_pl: c_labels,
                                })

        print(error / int(len(test_images) / batch_size))


        for k in [0, 200, 300, 400, 500, 600]:
            pred_angle = sess.run(q_output, 
                                feed_dict={
                                    states_pl: test_images[k, :, :].reshape([-1, 84, 84, 1]),
                                })
            print('predicted angle:', pred_angle)
            print('true angle:', test_labels[k])

        # for i in range(0, 6):
        #     image = images_n[i,:,:]
        #     #image = np.random.randint(255, size=84*84).reshape([84, 84]) 

        #     plt.imshow(image)
        #     plt.pause(1)

        #     pred = sess.run(prediction, feed_dict={
        #         states_pl: image.reshape(1,84,84,1)
        #     })

        #     print(pred)

        

        

run()