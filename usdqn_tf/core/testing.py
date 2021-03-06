import os
import sys
import time

import numpy as np
import tensorflow as tf

from core.utils import evaluate, reward_value, do_obs_processing, reward_clip

GAMMA = 0.99
TEST_STEPS = 100
FRAME_WIDTH = 80
FRAME_HEIGHT = 80
FRAME_BUFFER_SIZE = 1


def do_testing(env, model, target_model=None, dpaths=None, render=False, num_episodes=100):

    #print("Is on test mode ?", not env.is_training)
    tf.reset_default_graph()


    # Create placeholders
    states_pl = tf.placeholder(tf.float32, 
        shape=(None, FRAME_WIDTH, FRAME_HEIGHT, FRAME_BUFFER_SIZE), name='states')
    actions_pl= tf.placeholder(tf.int32, shape=(None), name='actions')
    targets_pl = tf.placeholder(tf.float32, shape=(None), name='targets')

    # Value function approximator network
    q_output = model.graph(states_pl)

    # Build target network
    q_target_net = target_model.graph(states_pl)

    # Compute Q from current q_output and one hot actions
    Q = tf.reduce_sum(
            tf.multiply(q_output, 
                tf.one_hot(actions_pl, env.action_space.n, dtype=tf.float32)
            ), axis=1)

    # Loss operation 
    loss_op = tf.reduce_mean(tf.square(targets_pl - Q) / 2)

    # Prediction Op
    prediction = tf.argmax(q_output, 1)
    #prediction = q_output

    # Model Saver
    saver = tf.train.Saver()

    # init all variables
    init_op = tf.global_variables_initializer()

    # Limit memory usage for multiple training at same time
    config = tf.ConfigProto(allow_soft_placement=True)
    #config.gpu_options.per_process_gpu_memory_fraction = 0.33

    # Start Session
    with tf.Session(config=config) as sess:

        if dpaths is not None:
            new_saver = tf.train.import_meta_graph(dpaths[1])
            new_saver.restore(sess, tf.train.latest_checkpoint(dpaths[0]))

            data_stats = None
            if env.usdqn_sim.data_testing:
                for i in range(len(env.usdqn_sim.wheel_data)):
                    print('i:%s/%s' % (i, len(env.usdqn_sim.wheel_data)))
                    means, stds, vmin, vmax = evaluate(env, sess, prediction, 
                                states_pl, env.usdqn_sim.data_testing, GAMMA, False, render)

                    stats = np.concatenate([np.array([i]), means, stds, vmin, vmax], axis=-1).reshape([1, -1])
                    if data_stats is None:
                        data_stats = stats
                    else:
                        data_stats = np.concatenate([data_stats, stats], axis=0)

                np.savetxt(os.path.join(dpaths[0], 'testing.csv'), data_stats, delimiter=',')

            else:
                means, stds = evaluate(env, sess, prediction, 
                                states_pl, num_episodes, GAMMA, False, render)

                #env.usdqn_sim.save_history()
            # Save means
            print(means)