import math
import time
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from envs.utils import digitize_indexes

from skimage.color import gray2rgb
#import matplotlib.pyplot as plt



RAD2DEG = 57.29577951308232

#f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# f2, (ax11, ax22) = plt.subplots(1, 2, sharey=True)

def is_sorted(l):
    return all(l[i] <= l[i+1] for i in range(len(l)-1))

class DiscretizedStateSpace(object):

    def __init__(self, step=0.02, is_training=True):
        super(DiscretizedStateSpace, self).__init__()
        self.dstep = step # Discretization step in radians
        self.is_training = is_training
        self.images, self.labels = None, None
        self.wheel_data, self.wheel_goal, self.wheel_goal =  None, None, None
        self.action_space_lim = None
        self.is_done = False
        self.cursor = 0
        self._load_dataset()

    def _load_dataset(self):
        if self.is_training:
            print("# Loading training set.")
            self.images = np.load('../data/1dof/usdqn-images-training.npy')[:, 2:-2, 2:-2]
            self.labels = np.load('../data/1dof/usdqn-labels-training.npy')
        else:
            print("# Loading testing set.")
            self.images = np.load('../data/1dof/usdqn-images-testing.npy')[:, 2:-2, 2:-2]
            self.labels = np.load('../data/1dof/usdqn-labels-testing.npy')

        # Data specific information
        data_min_angle = -3.05 #-175deg
        data_max_angle = 3.051 #175deg
        data_goal = [int((np.pi / 2) / self.dstep), int((np.pi / 2 + np.pi) / self.dstep)]

        n, bins = np.histogram(self.labels, 
            bins=np.arange(data_min_angle, data_max_angle, self.dstep))

        self.action_space_lim = int((np.pi * len(bins)) / (2*(data_max_angle - data_min_angle)))

        self.wheel_data = np.array(digitize_indexes(self.labels, bins))

        self.wheel_goal = np.zeros([len(self.wheel_data)])
        self.wheel_goal[data_goal] = 1

        #self.wheel = np.stack([wheel_indexes, wheel_goal], axis=1)

    def get_dist_to_goal(self, dangle):
        # if the first objective is in the first quadrant then it is the closest
        # otherwise the closest objective is in the 4 quadrant
        wheel_goal = np.roll(self.wheel_goal, -dangle)
        goals = np.argwhere(wheel_goal == 1)
        if goals[0][0] <= self.action_space_lim:
            return goals[0][0]
        else:
            return len(wheel_goal) - goals[1][0]

    def rotate_wheel(self, dangle, keep=False):
        
        if keep:
            # keep track of the rotation
            if abs(self.cursor) == len(self.wheel_goal):
                self.cursor = 0
            else:
                self.cursor += dangle
            dangle = self.cursor

        # print(len(self.wheel_goal))
        # print(dangle)
        # For rendering purposes
        self.wheel_angle = dangle

        # Sample random obs from current wheel box
        obs_ind = np.random.choice(self.wheel_data[dangle], 1)[0]

        if self.wheel_goal[dangle] == 1:
            self.is_done = True

        return (self.images[obs_ind, :, :], 
            self.wheel_goal[dangle], self.get_dist_to_goal(dangle))

    def reset_wheel(self):
        # Reset state space at random angle
        rnd_ind = np.random.randint(0, len(self.wheel_data))
        self.wheel_data = np.roll(self.wheel_data, rnd_ind, axis=0)
        self.wheel_goal = np.roll(self.wheel_goal, rnd_ind, axis=0)

        return self.rotate_wheel(0)

    def get_goal_angle(self):
        goal = np.argwhere(self.wheel_goal == 1)[0][0]
        return goal * self.dstep

    def get_angle(self):
        return self.wheel_angle * self.dstep


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

