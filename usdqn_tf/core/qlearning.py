import sys
import time

import numpy as np
import tensorflow as tf

from core.utils import evaluate, reward_value, do_obs_processing, reward_clip

GAMMA = 0.99
# TRAINING_STEPS = 500000
# LOG_STEPS = 1000
# LOSS_STEPS = 1000
# EVAL_STEPS = 25000
# SAVE_STEPS = 5000
# TARGET_UPDATE = 1000
TRAINING_STEPS = 1000000
LOG_STEPS = 5000
LOSS_STEPS = 1000
EVAL_STEPS = 50000
SAVE_STEPS = 50000
TARGET_UPDATE = 5000

FRAME_WIDTH = 84
FRAME_HEIGHT = 84
FRAME_BUFFER_SIZE = 4


def do_online_qlearning(env, 
                        test_env,    
                        model, 
                        learning_rate, 
                        epsilon_s,
                        gpu_device,
                        target_model=None, 
                        replay_buffer=None,
                        dpaths=None,
                        training=True):

    tf.reset_default_graph()

    with tf.device(gpu_device):

        # Create placeholders
        states_pl = tf.placeholder(tf.float32, 
            shape=(None, FRAME_WIDTH, FRAME_HEIGHT, FRAME_BUFFER_SIZE), name='states')
        actions_pl= tf.placeholder(tf.float32, shape=(None), name='actions')
        targets_pl = tf.placeholder(tf.float32, shape=(None), name='targets')

        # Value function approximator network
        q_output, q_debug = model.graph(states_pl)

        # Build target network
        q_target_net, q_targ_debug = target_model.graph(states_pl)

        tf_train = tf.trainable_variables()
        num_tf_train = len(tf_train)
        target_net_vars = []
        for i, var in enumerate(tf_train[0:num_tf_train // 2]):
            target_net_vars.append(tf_train[i + num_tf_train // 2].assign(var.value()))

        # Compute Q from current q_output and one hot actions
        # Q = tf.reduce_sum(
        #         tf.multiply(q_output, actions_pl), axis=1)

        # Loss operation 
        loss_op = tf.reduce_mean(tf.square(targets_pl - q_output) / 2)

        # Optimizer Op
        #optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # Training Op
        train_op = optimizer.minimize(loss_op)

        # Prediction Op
        #prediction = tf.argmax(q_output, 1)
        prediction = q_output

    # Model Saver
    saver = tf.train.Saver()

    # init all variables
    init_op = tf.global_variables_initializer()

    # Limit memory usage for multiple training at same time
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.33

    # Start Session
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        sess.graph.finalize() 

        # Performance from untrained Q-learning
        if not training:
            return evaluate(test_env, sess, prediction,
                                states_pl, 1, GAMMA, False)

        start_time = time.time()

        # Load env and get observations
        observation = env.reset()

        # Observation Buffer
        observation_buffer = list()

        # Save results
        losses = list()
        means = list()

        #init epsilon
        epsilon = epsilon_s['start']

        for step in range(TRAINING_STEPS):
            print("## Step :", step)
            loss = 0
            time_chkpt = time.time()
            # Stack observations in buffer of 4
            if len(observation_buffer) < FRAME_BUFFER_SIZE:

                observation_buffer.append(
                    do_obs_processing(observation, FRAME_WIDTH, FRAME_HEIGHT))

                # Collect next observation with uniformly random action
                a_rnd = env.action_space.sample()
                observation, _, done, _ = env.step(a_rnd)
                print("#1 ", time.time() - time_chkpt)
                time_chkpt = time.time()
            # Observations buffer is ready
            else:
                # Stack observation buffer
                state = np.stack(observation_buffer, axis=-1)

                print("#2 ", time.time() - time_chkpt)
                time_chkpt = time.time()
                q_deb= None
                # Epsilon greedy policy
                if epsilon > np.random.rand(1):
                    # Exploration
                    # Use uniformly sampled action from env
                    action = np.array(env.action_space.sample(), dtype=np.float32).reshape((-1))
                else:
                    # Exploitation 
                    # Use model predicted action 
                    action, q_deb = sess.run([prediction, q_debug], feed_dict={
                        states_pl: state.reshape(
                            [-1, FRAME_WIDTH, FRAME_HEIGHT, FRAME_BUFFER_SIZE]).astype('float32')
                    })

                print("#3 ", time.time() - time_chkpt)
                time_chkpt = time.time()
                # print('action taken:', action)
                # print('q debug:', q_deb)

                # action for next observation
                observation, reward, done, info  = env.step(action[0])

                # Clip reward
                r = reward_clip(reward)

                # Update observation buffer
                observation_buffer.append(
                    do_obs_processing(observation, FRAME_WIDTH, FRAME_HEIGHT))
                observation_buffer[0:1] = []

                print("#4 ", time.time() - time_chkpt)
                time_chkpt = time.time()

                next_state = np.stack(observation_buffer, axis=-1)
                action = action.reshape([-1]).astype('int32')

                # Add transition to replay buffer
                replay_buffer.add((state, action, r, next_state, done))

                print("#5 ", time.time() - time_chkpt)
                time_chkpt = time.time()
                # If replay buffer is ready to be sampled
                if replay_buffer.ready:
                    # print('train')
                    # Train model on replay buffer
                    b_states, b_actions, b_reward, b_next_state, b_term_state = replay_buffer.next_transitions()

                    # Run training on batch
                    q_out = sess.run(q_target_net, feed_dict={
                            states_pl: b_next_state
                        })
                    q_out_max = np.amax(q_out, axis=1)
                    q_target = b_reward + GAMMA * (1 - b_term_state) * q_out_max
                    print("#6 ", time.time() - time_chkpt)
                    time_chkpt = time.time()

                    # Run training Op on batch of replay experience
                    loss, _ = sess.run([loss_op, train_op], 
                        feed_dict={
                            states_pl: b_states,
                            actions_pl: b_actions,
                            targets_pl: q_target.astype('float32')
                        })
                    print("#7 ", time.time() - time_chkpt)
                    time_chkpt = time.time()
    
            if done:
                observation = env.reset()

            # Update epsilon greedy policy
            if epsilon > epsilon_s['end']:
                epsilon -= (epsilon_s['start'] - epsilon_s['end']) / epsilon_s['decay']

            # Copy variables target network
            if (step + 1) % TARGET_UPDATE == 0:
                for var in target_net_vars:
                    sess.run(var)

            print("#8 ", time.time() - time_chkpt)
            time_chkpt = time.time()

            if step % LOSS_STEPS == 0:
                # Save loss
                losses.append(loss)

            print("#9 ", time.time() - time_chkpt)
            time_chkpt = time.time()

            if step % LOG_STEPS == 0:
                print('\n', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
                print('Episodes %d -> %d Done (%.3fs) ... ' % 
                    (max(1, step + 1 - LOG_STEPS), step+1, time.time() - start_time))
                print('- Training loss: %.4f' % loss)
                start_time = time.time()

                # Force flush for nohup
                sys.stdout.flush()

            print("#10 ", time.time() - time_chkpt)
            time_chkpt = time.time()

            if step % EVAL_STEPS == 0:
                silent = (step % LOG_STEPS != 0)
                cur_means, cur_stds = evaluate(test_env, sess, prediction, 
                                        states_pl, 5, GAMMA, silent)

                # Save means
                means.append(cur_means)

            print("#11 ", time.time() - time_chkpt)
            time_chkpt = time.time()

            # Save models
            if dpaths is not None and step % SAVE_STEPS == 0:
                saver.save(sess, dpaths, global_step=step)

            print("#12 ", time.time() - time_chkpt)
            time_chkpt = time.time()
            
            sys.stdout.flush()

        # Save models
        if dpaths is not None:
            saver.save(sess, dpaths)

    # Return Q-learning Experience results
    return losses, means