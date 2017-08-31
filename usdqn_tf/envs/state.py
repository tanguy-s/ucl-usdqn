import numpy as np
from envs.utils import digitize_indexes

#import matplotlib.pyplot as plt

class DiscretizedStateSpace(object):

    def __init__(self, step=0.02, is_training=True, sample=True):
        super(DiscretizedStateSpace, self).__init__()
        self.dstep = step # Discretization step in radians
        self.sample = sample
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
            self.cursor += dangle
            dangle = self.cursor

        # For rendering purposes
        self.wheel_angle = dangle

        # Sample random obs from current wheel box
        wheel_data = self.wheel_data.take(dangle, mode='wrap') 
        wheel_goal = self.wheel_goal.take(dangle, mode='wrap') 

        if self.sample:
            # sample from the bin of images
            obs_ind = np.random.choice(wheel_data, 1)[0]
        else:
            # always take the same image
            obs_ind = wheel_data[0]

        if wheel_goal == 1:
            self.is_done = True
            # plt.imshow(self.images[obs_ind, :, :])
            # plt.pause(0.01)

        return (self.images[obs_ind, :, :], 
            wheel_goal, self.get_dist_to_goal(dangle))

    def reset_wheel(self):
        # Reset state space at random angle
        self.is_done = False
        self.cursor = 0
        rnd_ind = np.random.randint(0, len(self.wheel_data))
        self.wheel_data = np.roll(self.wheel_data, rnd_ind, axis=0)
        self.wheel_goal = np.roll(self.wheel_goal, rnd_ind, axis=0)

        return self.rotate_wheel(0)

    def get_goal_angle(self):
        goal = np.argwhere(self.wheel_goal == 1)[0][0]
        return goal * self.dstep

    def get_angle(self):
        return self.wheel_angle * self.dstep