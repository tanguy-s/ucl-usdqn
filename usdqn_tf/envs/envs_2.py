import math
import time
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from skimage.color import gray2rgb
#import matplotlib.pyplot as plt



RAD2DEG = 57.29577951308232

# f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# f2, (ax11, ax22) = plt.subplots(1, 2, sharey=True)

def is_sorted(l):
    return all(l[i] <= l[i+1] for i in range(len(l)-1))

class UsdqnOneDoFSimulator(object):

    def __init__(self, is_training=True):
        self.is_training = is_training
        self.min_action = -3.05
        #self.max_action = 3.05
        self.max_action = -3.05 + 180*(1/RAD2DEG)

        # self.discrete_actions = np.arange(self.min_action, 
        #     self.max_action, 0.01)

        self.discrete_actions = np.arange(self.min_action, 
            self.max_action, 0.01)

        #print("SIZE of actions:", len(self.discrete_actions))

        self.goal_positions = None
        self.current_indx = None
        self.goal_indx = None

        self.load_dataset()

    def load_dataset(self):
        if self.is_training:
            self.images = np.load('../data/1dof/usdqn-images-training.npy')
            self.labels = np.load('../data/1dof/usdqn-labels-training.npy')
            
        else:
            self.images = np.load('../data/1dof/usdqn-images-testing.npy')
            self.labels = np.load('../data/1dof/usdqn-labels-testing.npy')
            # self.goal_positions = np.array([
            #     -1.48979521 
            # ])

        end_indx = (np.abs(self.labels - self.max_action)).argmin()
        # print("END INDX:", end_indx)
        self.images = self.images[0:end_indx,2:-2,2:-2]
        self.labels = self.labels[0:end_indx]
        print("IMGS SHAPE:", self.images.shape)
        # print("LABELS SHAPE:", self.labels.shape)
        for im in range(0, self.images.shape[0]):
            print()



        self.goal_positions = np.array([
                self.min_action + 90*(1/RAD2DEG),
                #self.min_action + 270*(1/RAD2DEG)
            ])

        self.goal_indx = (np.abs(self.labels - self.goal_positions[0])).argmin()
        # print("INIT GOAL INDEX:", self.goal_indx)
        # ax11.imshow(self.images[self.goal_indx,:,:])
        # f2.show()

    def set_angle(self, action):
        print("Action: ", action)
        # Find nearest observation in array
        action = self.discrete_actions[action]
        self.current_indx = (np.abs(self.labels - action)).argmin()
        
    def is_done(self):
        #print(self.labels[self.current_indx])
        # if self.current_indx == self.goal_indx:
        #     print("## Done objective:")
        #     return True
        # else:
        #     return False

        if np.any(np.abs(self.goal_positions - self.labels[self.current_indx]) < 0.01):
            print("## Done objective: %s" % np.abs(self.goal_positions - self.labels[self.current_indx]))
        return np.any(
            (np.abs(self.goal_positions - self.labels[self.current_indx]) < 0.01))

    # def reset(self, np_random):
    #     print("## Reseting")
    #     ridx = np_random.uniform(low=self.min_action, high=self.max_action)
    #     #print("RESETING: %s" % ridx)
    #     self.current_indx = (np.abs(self.labels - ridx)).argmin()

    def reset(self, np_random):
        # Random angle in range 0->175
        rnd_angle = np_random.uniform(low=0, high=180*(1/RAD2DEG))
        #print("## Reseting")
        #print("- New random angle:", rnd_angle)
        #print("- Are labels sorted ?", is_sorted(self.labels))
        self.labels += rnd_angle
        self.goal_positions += rnd_angle
        if np.any(self.goal_positions > self.max_action):
            goal_ind = np.where(self.goal_positions > self.max_action)[0]
            self.goal_positions[goal_ind] -= (self.max_action - self.min_action)
        #print("- Goal positions:", self.goal_positions)
        over_ind = np.where(self.labels > self.max_action)[0]
        #print("- Over indexes:", over_ind)
        self.labels[over_ind] -= (self.max_action - self.min_action)
        to_shift = len(over_ind)
        #print("- To shift:", to_shift)
        # Roll images and labels
        self.labels = np.roll(self.labels, to_shift)
        self.images = np.roll(self.images, to_shift, axis=0) 

        #print("- Are labels still sorted ?", is_sorted(self.labels))
        #self.current_indx = int(len(self.labels) / 2)
        self.current_indx = np.random.choice(len(self.labels), 1)[0]
        #print("NEW CURRENT INDEX:", self.current_indx)


    def get_image(self):
        if self.current_indx is None:
            raise 
        # goal_indx = (np.abs(self.labels - self.goal_positions[0])).argmin()
        # ax1.imshow(self.images[self.current_indx, :, :])
        # ax2.imshow(self.images[goal_indx,:,:])
        # plt.pause(0.0001)
        return self.images[self.current_indx, :, :]

    def get_reward(self):
        if self.current_indx is None:
            raise
        if self.is_done():
            return 0
        return -1

    def get_angle(self):
        if self.current_indx is None:
            raise 
        return self.labels[self.current_indx]


class Continuous_UsdqnOneDoFEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self, is_training=True):
        self.is_training = is_training

        self.usdqn_sim = UsdqnOneDoFSimulator(is_training)

        self.viewer = None

        #self.action_space = spaces.Box(self.usdqn_sim.min_action, 
        #    self.usdqn_sim.max_action, shape=(1,))
        self.action_space = spaces.Discrete(len(self.usdqn_sim.discrete_actions))
        self.observation_space = spaces.Box(low=0, high=1, shape=(80, 80, 1))

        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        #assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        
        self.usdqn_sim.set_angle(action)

        state = self._get_obs()
        reward = self.usdqn_sim.get_reward()
        #print("- Reward: ", reward)
        done = self.usdqn_sim.is_done()
        #print("- Done: ", done)
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

        img = self.usdqn_sim.get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'none':
            from envs.rendering import GrayImageViewer
            # #
            if self.viewer is None:
                self.viewer = GrayImageViewer()
            self.viewer.imshow(img)

        elif mode == 'human':
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
                self.needle_transform.set_rotation(self.usdqn_sim.goal_positions[0])
                needle_plane.add_attr(self.needle_transform)
                self.viewer.add_geom(needle_plane)

                self.img = ImageData(img, 84, 84)
                self.imgtrans = rendering.Transform()
                self.img.add_attr(self.imgtrans)

                #self.viewer.add_geom(self.img)

                self.us_plane = rendering.make_polygon([(-0.01, -1), (-0.01, 1), (0.01, 1), (0.01, -1)], False)
                self.us_plane.set_color(.8, .3, .3)
                self.us_plane.set_linewidth(2)
                self.us_transform = rendering.Transform()
                self.us_plane.add_attr(self.us_transform)
                self.viewer.add_geom(self.us_plane)

            time.sleep(0.1)
            if self.usdqn_sim.is_done():
                self.us_plane.set_color(50/255,205/255,50/255)
                plt.imshow(img, animated=True)
                plt.pause(0.05)
            else:
                self.us_plane.set_color(.8, .3, .3)
            self.needle_transform.set_rotation(self.usdqn_sim.goal_positions[0])
            self.us_transform.set_rotation(self.usdqn_sim.get_angle())
            #self.capture.capture()

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

            # plt.imshow(img, animated=True)
            # plt.pause(0.05)

            #imgplot = plt.imshow(img)
            #plt.show()

# Evaluation of policy
# - Return stats:
#  Mean: -75.558822 std: 37.779411
# - Scores stats:
#  Mean: -396.800000 std: 198.400000

# Evaluation of policy
# - Return stats:
#  Mean: -94.448527 std: 0.000000
# - Scores stats:
#  Mean: -496.000000 std: 0.000000
