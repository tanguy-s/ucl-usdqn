import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from skimage.color import gray2rgb
import matplotlib.pyplot as plt


RAD2DEG = 57.29577951308232

class UsdqnOneDoFSimulator(object):

    def __init__(self, is_training=True):
        self.is_training = is_training
        self.min_action = -3.05
        self.max_action = 3.05

        self.discrete_actions = np.arange(self.min_action, 
            self.max_action, 0.01)

        self.goal_positions = [
            self.min_action + 90*(1/RAD2DEG),
            self.min_action + 270*(1/RAD2DEG)
        ]

        self.current_indx = None

        self.load_dataset()

    def load_dataset(self):
        if self.is_training:
            self.images = np.load('../data/1dof/usdqn-images-training.npy')
            self.labels = np.load('../data/1dof/usdqn-labels-training.npy')
        else:
            self.images = np.load('../data/1dof/usdqn-images-testing.npy')
            self.labels = np.load('../data/1dof/usdqn-labels-testing.npy')

    def set_angle(self, action):
        # Find nearest observation in array
        action = self.discrete_actions[action]
        self.current_indx = (np.abs(self.labels - action)).argmin()

    def is_done(self):
        #print(self.labels[self.current_indx])
        if np.any(np.abs(self.goal_positions - self.labels[self.current_indx]) < 0.01):
            print("######################")
            print("Is done objective: %s" % np.abs(self.goal_positions - self.labels[self.current_indx]))
        return np.any(
            (np.abs(self.goal_positions - self.labels[self.current_indx]) < 0.001))

    def reset(self, np_random):
        ridx = np_random.uniform(low=self.min_action, high=self.max_action)
        print("RESETING: %s" % ridx)
        self.current_indx = (np.abs(self.labels - ridx)).argmin()

    def get_image(self):
        if self.current_indx is None:
            raise 
        return self.images[self.current_indx, :, :]

    def get_reward(self):
        if self.current_indx is None:
            raise
        return -1


class Continuous_UsdqnOneDoFEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self, is_training=True):
        self.is_training = is_training

        self.usdqn_sim = UsdqnOneDoFSimulator(is_training)

        self.viewer = None

        # self.action_space = spaces.Box(self.usdqn_sim.min_action, 
        #     self.usdqn_sim.max_action, shape=(1,))
        self.action_space = spaces.Discrete(len(self.usdqn_sim.discrete_actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1))

        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        print(self.usdqn_sim.current_indx)
        
        self.usdqn_sim.set_angle(action)

        print(self.usdqn_sim.current_indx)

        state = self._get_obs()
        reward = self.usdqn_sim.get_reward()
        done = self.usdqn_sim.is_done()
        return state, reward, done, {}

    def _get_obs(self):
        return self.usdqn_sim.get_image()

    def _reset(self):
        self.usdqn_sim.reset(self.np_random)
        return self._get_obs()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        img = self.usdqn_sim.get_image().reshape([84, 84, 1]).astype('uint8')
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from envs.rendering import GrayImageViewer
            # #
            if self.viewer is None:
                self.viewer = GrayImageViewer()
            self.viewer.imshow(img)

            # plt.imshow(img, animated=True)
            # plt.pause(0.05)

            #imgplot = plt.imshow(img)
            #plt.show()
