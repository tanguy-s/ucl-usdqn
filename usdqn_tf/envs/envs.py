import math
import time
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from envs.state import DiscretizedStateSpace

from skimage.color import gray2rgb
#import matplotlib.pyplot as plt



RAD2DEG = 57.29577951308232

#f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# f2, (ax11, ax22) = plt.subplots(1, 2, sharey=True)

def is_sorted(l):
    return all(l[i] <= l[i+1] for i in range(len(l)-1))




class OneDoFSim_BinaryActions_Supervised(DiscretizedStateSpace):

    def __init__(self, step=0.02, is_training=False):
        super(OneDoFSim_BinaryActions_Supervised, self).__init__(step, is_training)
        self.actions = np.array([-1,1])
        self.dist_to_goal = None

    def reset(self):
        step = self.reset_wheel()
        self.dist_to_goal = step[2]
        return step[0]

    def step(self, action):
        action = self.actions[action]
        step = self.rotate_wheel(action, keep=True)
        if self.dist_to_goal > step[2]:
            reward = -1
        else:
            reward = 0
        return step[0], reward, bool(step[1]), {}


class OneDoFSim_BinaryActions_Unsupervised(DiscretizedStateSpace):

    def __init__(self, step=0.02, is_training=False):
        super(OneDoFSim_BinaryActions_Unsupervised, self).__init__(step, is_training)
        self.actions = np.array([-1,1])

    def reset(self):
        step = self.reset_wheel()
        return step[0]

    def step(self, action):
        action = self.actions[action]
        step = self.rotate_wheel(action, keep=True)
        reward = 0 if step[1] == 1 else -1
        return step[0], reward, bool(step[1]), {}


class OneDoFSim_DiscActions_Unsupervised(DiscretizedStateSpace):

    def __init__(self, step=0.02, is_training=False):
        super(OneDoFSim_DiscActions_Unsupervised, self).__init__(step, is_training)
        self.actions = np.arange(-self.action_space_lim, self.action_space_lim, 1)

    def reset(self):
        step = self.reset_wheel()
        return step[0]

    def step(self, action):
        action = self.actions[action]
        step = self.rotate_wheel(action)
        reward = 0 if step[1] == 1 else -1
        return step[0], reward, bool(step[1]), {}


class OneDoFSim_DiscActions_Supervised(DiscretizedStateSpace):

    def __init__(self, step=0.02, is_training=False):
        super(OneDoFSim_DiscActions_Supervised, self).__init__(step, is_training)
        self.actions = np.arange(-self.action_space_lim, self.action_space_lim, 1)

    def reset(self):
        step = self.reset_wheel()
        return step[0]

    def step(self, action):
        action = self.actions[action]
        step = self.rotate_wheel(action)
        reward = -step[2]
        #print("dist:", step[2])
        #print("reward:", reward)
        return step[0], reward, bool(step[1]), {}



class UsdqnOneDoFEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self, simulator):
        self.usdqn_sim = simulator

        self.viewer = None

        self.action_space = spaces.Discrete(len(self.usdqn_sim.actions))
        self.observation_space = spaces.Box(low=0, high=1, shape=(80, 80, 1))

        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        return self.usdqn_sim.step(action)

    def _reset(self):
        return self.usdqn_sim.reset()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        # img = self.usdqn_sim.get_image()
        # if mode == 'rgb_array':
        #     return img
        # elif mode == 'none':
        #     from envs.rendering import GrayImageViewer
        #     # #
        #     if self.viewer is None:
        #         self.viewer = GrayImageViewer()
        #     self.viewer.imshow(img)

        if mode == 'human':
            if self.viewer is None:
                from gym.envs.classic_control import rendering
                from envs.rendering import ImageData, MovieCapture

                self.viewer = rendering.Viewer(500,500)
                self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
                #self.capture = MovieCapture('./dumps/video/test')

                needle_plane = rendering.Line((0, -1), (0, 1))
                needle_plane.set_color(68/255,108/255,179/255)
                needle_plane.linewidth = 4
                needle_plane.add_attr(rendering.LineWidth(4))
                self.needle_transform = rendering.Transform()
                self.needle_transform.set_rotation(self.usdqn_sim.get_goal_angle())
                needle_plane.add_attr(self.needle_transform)
                self.viewer.add_geom(needle_plane)

                # self.img = ImageData(img, 84, 84)
                # self.imgtrans = rendering.Transform()
                # self.img.add_attr(self.imgtrans)

                #self.viewer.add_geom(self.img)

                self.us_plane = rendering.make_polygon([(-0.01, -1), (-0.01, 1), (0.01, 1), (0.01, -1)], False)
                self.us_plane.set_color(.8, .3, .3)
                self.us_plane.set_linewidth(2)
                self.us_transform = rendering.Transform()
                self.us_plane.add_attr(self.us_transform)
                self.viewer.add_geom(self.us_plane)

            time.sleep(0.1)
            if self.usdqn_sim.is_done:
                self.us_plane.set_color(50/255,205/255,50/255)
            else:
                self.us_plane.set_color(.8, .3, .3)
            self.needle_transform.set_rotation(self.usdqn_sim.get_goal_angle())
            self.us_transform.set_rotation(self.usdqn_sim.get_angle())
            #self.capture.capture()

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

