import warnings
import time
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

#import matplotlib.pyplot as plt


FRAME_WIDTH = 80
FRAME_HEIGHT = 80
FRAME_BUFFER_SIZE = 1
MAX_EPISODE = 700


def reward_clip(reward):
    #return np.clip(reward, -1, 1)
    return reward

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

        while not done:
            t += 1

            a_rnd = env.action_space.sample()
            #print("A random:", a_rnd)
            observation, reward, done, _ = env.step(a_rnd)

            # Clip reward
            #r = reward_clip(reward)

            # Return value
            retval += pow(gamma, t)*reward
            score += reward

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
                #q_output,
                num_episodes, 
                gamma, 
                silent=False,
                render=False):

    print('Evaluation num episodes: ', num_episodes)
    timings = list()

    res = np.zeros([num_episodes, 3])
    for i_episode in range(num_episodes):
        # if not silent:
        #     print('Episode %d/%d' % (i_episode, num_episodes))
        #Observation Buffer
        observation_buffer = list()

        # Reset environement variables
        observation = env.reset()
        observation_buffer.append(observation)
        done = False
        retval = 0 # Value function
        score = 0
        t = 0
        #f, axarr = plt.subplots(2, 2)

        while not done and t < MAX_EPISODE:
            t += 1

            
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

                start_time = time.time()
                action = sess.run(prediction_op, feed_dict={
                    states_pl: state.reshape(
                        [-1, FRAME_WIDTH, FRAME_HEIGHT, FRAME_BUFFER_SIZE]).astype('float32')
                })
                end_time = time.time() - start_time
                timings.append(end_time)
                # action for next observation
                observation, reward, done, info  = env.step(action[0])

                # Clip reward
                r = reward_clip(reward)

                # Return value
                retval += pow(gamma, t)*r
                score += r

                observation_buffer.append(observation)
                observation_buffer[0:1] = []

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
    vmin = np.min(res, axis=0)
    vmax = np.max(res, axis=0)

    if not silent:
        print('# Evaluation of policy')
        print('- Return stats:\n Mean: %f std: %f' % 
            (means[0], stds[0]))
        print('- Scores stats:\n Mean: %f std: %f' % 
            (means[1], stds[1]))
        print('- Frames stats:\n Mean: %f std: %f' % 
            (means[2], stds[2]))
        print('- timing is: %s std:%s' % (np.mean(timings[10:]), np.std(timings[10:])))

    return means, stds#, vmin, vmax


def do_obs_processing(frame, width, height):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        #return resize(frame, (110, 84))[13:110 - 13, :]
        return resize(frame, (width, height))


