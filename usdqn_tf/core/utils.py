import warnings
import time
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

#import matplotlib.pyplot as plt


FRAME_WIDTH = 28
FRAME_HEIGHT = 28
FRAME_BUFFER_SIZE = 4
MAX_EPISODE = 500


def reward_clip(reward):
    return np.clip(reward, -1, 1)

def reward_value(done):
    # Reward function: 0 in all states except 
    # if episode is done -1
    return -1 if done else 0

def evaluate_random(env,
                num_episodes, 
                gamma, 
                silent=False,
                render=False):
    
    res = np.zeros([num_episodes, 3])
    for i_episode in range(num_episodes):
        if not silent:
            print('Episode %d/%d' % (i_episode, num_episodes))

        # Reset environement variables
        observation = env.reset()
        if render:
            time.sleep(0.5)
        done = False
        retval = 0 # Value function
        score = 0
        t = 0

        while not done and t < MAX_EPISODE:
            t += 1

            a_rnd = env.action_space.sample()
            observation, reward, done, _ = env.step(a_rnd)

            # Clip reward
            r = reward_clip(reward)

            # Return value
            retval += pow(gamma, t)*r
            score += r

            if render:         
                env.render()

        res[i_episode][0] = retval
        res[i_episode][1] = score
        res[i_episode][2] = t

        if render:
            time.sleep(1)
            #break

    means = np.mean(res, axis=0)
    stds = np.std(res, axis=0)

    if not silent:
        print('# Evaluation of random policy')
        print('- Return stats:\n Mean: %f std: %f' % 
            (means[0], stds[0]))
        print('- Scores stats:\n Mean: %f std: %f' % 
            (means[1], stds[1]))
        print('- Frames stats:\n Mean: %f std: %f' % 
            (means[2], stds[2]))

    return means, stds

def evaluate(env,
                sess, 
                prediction_op, 
                states_pl, 
                num_episodes, 
                gamma, 
                silent=False,
                render=False):

    print('Evaluation num episodes: ', num_episodes)

    res = np.zeros([num_episodes, 2])
    for i_episode in range(num_episodes):
        if not silent:
            print('Episode %d/%d' % (i_episode, num_episodes))
        #Observation Buffer
        observation_buffer = list()

        # Reset environement variables
        observation = env.reset()
        done = False
        retval = 0 # Value function
        score = 0
        t = 0

        #f, axarr = plt.subplots(2, 2)

        while not done and t < MAX_EPISODE:
            t += 1
            #print("eval step:", t)
            
            # Stack observations in buffer of 4
            if len(observation_buffer) < FRAME_BUFFER_SIZE:

                observation_buffer.append(
                    do_obs_processing(observation, FRAME_WIDTH, FRAME_HEIGHT))

                # Collect next observation with uniformly random action
                #a_rnd = env.action_space.sample()
                #observation, _, done, _ = env.step(a_rnd)


            # Observations buffer is ready
            else:
                # Stack observation buffer
                state = np.stack(observation_buffer, axis=-1)

                action = sess.run(prediction_op, feed_dict={
                    states_pl: state.reshape(
                        [-1, FRAME_WIDTH, FRAME_HEIGHT, FRAME_BUFFER_SIZE]).astype('float32')
                })

                # q_test = sess.run(q_output, feed_dict={
                #     states_pl: state.reshape(
                #         [-1, FRAME_WIDTH, FRAME_HEIGHT, FRAME_BUFFER_SIZE]).astype('float32')
                # })

                # print("Q output:", q_test.shape)
                # print(q_test)

                #print("Action:", action[0])
                # action for next observation
                observation, reward, done, info  = env.step(action[0])

                
                # axarr[0, 0].imshow(observation_buffer[0])
                # # axarr[0, 1].imshow(observation_buffer[1])
                # # axarr[1, 0].imshow(observation_buffer[2])
                # # axarr[1, 1].imshow(observation_buffer[3])
                # plt.pause(5)



                # Clip reward
                r = reward_clip(reward)

                # Return value
                retval += pow(gamma, t)*r
                score += r

                observation_buffer.append(
                    do_obs_processing(observation, FRAME_WIDTH, FRAME_HEIGHT))
                observation_buffer[0:1] = []

            if render:         
                env.render()

        res[i_episode][0] = retval
        res[i_episode][1] = score

        if render:
            time.sleep(1)
            #break

    means = np.mean(res, axis=0)
    stds = np.std(res, axis=0)

    if not silent:
        print('# Evaluation of policy')
        print('- Return stats:\n Mean: %f std: %f' % 
            (means[0], stds[0]))
        print('- Scores stats:\n Mean: %f std: %f' % 
            (means[1], stds[1]))

    return means, stds


def do_obs_processing(frame, width, height):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        #return resize(frame, (110, 84))[13:110 - 13, :]
        return resize(frame, (width, height))

