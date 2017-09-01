import sys
import time

import numpy as np
import tensorflow as tf

from core.utils import evaluate, reward_value, do_obs_processing, reward_clip


# import matplotlib.pyplot as plt
#f, axarr = plt.subplots(2, 2)

# f, (ax1, ax2) = plt.subplots(1, 2)

GAMMA = 0.99
FRAME_WIDTH = 80
FRAME_HEIGHT = 80
FRAME_BUFFER_SIZE = 1



def do_online_qlearning(env, 
                        test_env,    
                        model, 
                        params,
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
        actions_pl= tf.placeholder(tf.int32, shape=(None), name='actions')
        targets_pl = tf.placeholder(tf.float32, shape=(None), name='targets')

        # Value function approximator network
        q_output = model.graph(states_pl)

        # Build target network
        q_target_net = target_model.graph(states_pl)

        tf_train = tf.trainable_variables()
        num_tf_train = len(tf_train)
        target_net_vars = []
        for i, var in enumerate(tf_train[0:num_tf_train // 2]):
            target_net_vars.append(tf_train[i + num_tf_train // 2].assign(var.value()))

        # Compute Q from current q_output and one hot actions
        # Q = tf.reduce_sum(
        #         tf.multiply(q_output, actions_pl), axis=1)

        Q = tf.reduce_sum(
                tf.multiply(q_output, 
                    tf.one_hot(actions_pl, env.action_space.n, dtype=tf.float32)
                ), axis=1)

        # Loss operation 
        loss_op = tf.reduce_mean(tf.square(targets_pl - Q) / 2)

        # Loss operation 
        #loss_op = tf.reduce_mean(tf.square(targets_pl - q_output) / 2)

        # Optimizer Op
        #optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # Training Op
        train_op = optimizer.minimize(loss_op)

        # Prediction Op
        prediction = tf.argmax(q_output, 1)
        #prediction = q_output

    # Model Saver
    saver = tf.train.Saver()

    # init all variables
    init_op = tf.global_variables_initializer()

    # Limit memory usage for multiple training at same time
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.49
    #config.gpu_options.allow_growth = True

    # Start Session
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        sess.graph.finalize() 

        # Performance from untrained Q-learning
        if not training:
            return evaluate(test_env, sess, prediction,
                                states_pl, params['EVAL_EPISODES'], GAMMA, False, False)

        start_time = time.time()

        # Load env and get observations
        observation = env.reset()

        # Observation Buffer
        observation_buffer = list()

        # Save results
        losses = list()
        means = list()
        stds = list()

        #init epsilon
        epsilon = epsilon_s['start']

        for step in range(params['TRAINING_STEPS']):
            loss = 0
            
            # Stack observations in buffer of 4
            if len(observation_buffer) < FRAME_BUFFER_SIZE:

                observation_buffer.append(observation)

                # Collect next observation with uniformly random action
                a_rnd = env.action_space.sample()
                observation, _, done, _ = env.step(a_rnd)
                
            # Observations buffer is ready
            else:
                # Stack observation buffer
                state = np.stack(observation_buffer, axis=-1)

                # Epsilon greedy policy
                if epsilon > np.random.rand(1):
                    # Exploration
                    # Use uniformly sampled action from env
                    action = np.array(env.action_space.sample(), dtype=np.float32).reshape((-1))
                else:
                    # Exploitation 
                    # Use model predicted action 
                    action = sess.run(prediction, feed_dict={
                        states_pl: state.reshape(
                            [-1, FRAME_WIDTH, FRAME_HEIGHT, FRAME_BUFFER_SIZE]).astype('float32') 
                    })
                
                # action for next observation
                action = action.reshape([-1]).astype('int32')
                observation, reward, done, info  = env.step(action[0])

                #print(type(observation))

                # Clip reward
                r = reward_clip(reward)

                # Update observation buffer
                observation_buffer.append(observation)
                observation_buffer[0:1] = []

                #print("max:", np.max(observation))

                next_state = np.stack(observation_buffer, axis=-1)

                #ax1.imshow(state[:,:,0])
                #ax2.imshow(next_state[:,:,0])
                # axarr[0, 0].imshow(b_next_state[1,:,:,0])
                # axarr[0, 1].imshow(b_next_state[1,:,:,1])
                # axarr[1, 0].imshow(b_next_state[1,:,:,2])
                # axarr[1, 1].imshow(b_next_state[1,:,:,3])

                #plt.pause(1)

                # axarr[0, 0].imshow(next_state[:,:,0])
                # axarr[0, 1].imshow(next_state[:,:,1])
                # axarr[1, 0].imshow(next_state[:,:,2])
                # axarr[1, 1].imshow(next_state[:,:,3])

                # plt.pause(0.001)

                # Add transition to replay buffer
                replay_buffer.add(((state * 255).astype('uint8'), action, r, (next_state * 255).astype('uint8'), done))
                
                # If replay buffer is ready to be sampled
                if replay_buffer.ready:
                    # Train model on replay buffer
                    b_states, b_actions, b_reward, b_next_state, b_term_state = replay_buffer.next_transitions()

                    #print(b_term_state)
                    #Run training on batch
                    # ax1.imshow(b_next_state[1,:,:,0])
                    # ax2.imshow(b_next_state[10,:,:,0])
                    # plt.pause(0.03)
                    # # axarr[0, 0].imshow(b_next_state[1,:,:,0])
                    # # axarr[0, 1].imshow(b_next_state[1,:,:,1])
                    # # axarr[1, 0].imshow(b_next_state[1,:,:,2])
                    # # axarr[1, 1].imshow(b_next_state[1,:,:,3])

                    # plt.pause(1)

                    # axarr[0, 0].imshow(b_next_state[4,:,:,0])
                    # axarr[0, 1].imshow(b_next_state[4,:,:,1])
                    # axarr[1, 0].imshow(b_next_state[4,:,:,2])
                    # axarr[1, 1].imshow(b_next_state[4,:,:,3])

                    # plt.pause(1)
                    # print("Mean:", np.mean(np.mean(np.mean(b_next_state, axis=3), axis=2), axis=1))
                    q_out, q_out_target = sess.run([q_output, q_target_net], feed_dict={
                            states_pl: b_next_state 
                        })

                    #print(np.max(b_next_state[1,:,:,:]))
                    # q_out1 = sess.run(q_output, feed_dict={
                    #           states_pl: b_next_state[1,:,:,:].reshape([1,80,80,1]) 
                    #       })

                    # q_out2 = sess.run(q_output, feed_dict={
                    #           states_pl: b_next_state[10,:,:,:].reshape([1,80,80,1])  
                    #       })
                    # print("Q out 1: ", np.argmax(q_out1))
                    # print("Q out 2: ", np.argmax(q_out2))
                    #print(b_next_state.shape)
                    #print(q_out)
                    #print(q_out_target.shape)
                    # q_out = sess.run(q_output, feed_dict={
                    #          states_pl: b_next_state 
                    #      })
                    #q_out_max = np.amax(q_out, axis=1)
                    #q_target = b_reward + GAMMA * (1 - b_term_state) * q_out_max

                    #q_target = b_reward + GAMMA * (1 - b_term_state) * q_out_max


                    #q_out_max = np.amax(q_out, axis=1)
                    #print(q_out_max)
                    
                    q_out_argmax = np.unravel_index(np.argmax(q_out, axis=1), q_out.shape)
                    #print(q_out_argmax)
                    #print(q_out_target[q_out_argmax])
                    # print("Q_out_target_max:", q_out_target[:, q_out_argmax])
                    # print(q_out_argmax)
                    #q_target = b_reward + GAMMA * (1 - b_term_state) * q_out_target[q_out_argmax]
                    #print(q_out_argmax.reshape([-1,1]))
                    # print("Target:", q_out_target[q_out_argmax])
                    # print("Cn:", q_out_max)

                    q_target = b_reward + GAMMA * (1 - b_term_state) * q_out_target[q_out_argmax]



                    # Run training Op on batch of replay experience
                    loss, _ = sess.run([loss_op, train_op], 
                        feed_dict={
                            states_pl: b_states,
                            actions_pl: b_actions,
                            targets_pl: q_target.astype('float32')
                        })
    
                
    
            if done:
                observation = env.reset()

            # Update epsilon greedy policy
            if epsilon > epsilon_s['end']:
                epsilon -= (epsilon_s['start'] - epsilon_s['end']) / epsilon_s['decay']

            # Copy variables target network
            if (step + 1) % params['TARGET_UPDATE'] == 0:
                for var in target_net_vars:
                    sess.run(var)
            

            if step % params['LOSS_STEPS'] == 0:
                # Save loss
                losses.append(loss)
            

            if step % params['LOG_STEPS'] == 0:
                print('\n', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
                print('Episodes %d -> %d Done (%.3fs) ... ' % 
                    (max(1, step + 1 - params['LOG_STEPS']), step+1, time.time() - start_time))
                print('- Training loss: %.4f' % loss)
                start_time = time.time()

                # Force flush for nohup
                sys.stdout.flush()
            

            if 'EVAL_STEPS_START' in params:
                if (step % params['EVAL_STEPS_START'] == 0 
                    and step <= params['EVAL_STEPS_STOP'] and step != 0):

                    silent = (step % params['LOG_STEPS'] != 0)
                    cur_means, cur_stds = evaluate(test_env, sess, prediction, 
                                        #states_pl, 1, GAMMA, silent)
                                        states_pl, params['EVAL_EPISODES'], GAMMA, silent)

                    means.append(cur_means)
                    stds.append(cur_stds)

            if step % params['EVAL_STEPS'] == 0:
                silent = (step % params['LOG_STEPS'] != 0)
                cur_means, cur_stds = evaluate(test_env, sess, prediction, 
                                        #states_pl, 1, GAMMA, silent)
                                        states_pl, params['EVAL_EPISODES'], GAMMA, silent)

                # Save means
                means.append(cur_means)
                stds.append(cur_stds)
            

            # Save models
            if dpaths is not None and step % params['SAVE_STEPS'] == 0:
                saver.save(sess, dpaths, global_step=step)
            

        # Save models
        if dpaths is not None:
            saver.save(sess, dpaths)

    # Return Q-learning Experience results
    return losses, means, stds