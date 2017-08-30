import warnings
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt

from PIL import Image

import scipy.misc
import matplotlib
import matplotlib.cm as cm

def do_obs_processing(frame, width, height):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return resize(frame, (width, height))

images = np.load('../data_raw/1dof/raw/usdqn-images-1.npy')
# for k in range(2,9):
#     print(k)
#     imgs = np.load('../data_raw/1dof/raw/usdqn-images-%s.npy' % k)
#     images = np.concatenate([images, imgs], axis=0)

print(images.shape)

RAD2DEG = 57.29577951308232

images = images.reshape(-1, 616, 820)

labels = np.load('../data_raw/1dof/raw/usdqn-labels.npy')

labels_training = np.load('./data/1dof/usdqn-labels-training.npy')
labels_test = np.load('./data/1dof/usdqn-labels-testing.npy')

# print(len(labels_test))
# print(labels_test)

# n, bins = np.histogram(labels_training, bins=np.arange(-3.05, 3.051, 0.02))
# print(n)
# print(np.sum(n))
# ind = np.digitize(labels_training, bins)
# print(ind)
# print("Num bins:", len(bins))

# data_min_angle = -3.05 #-175deg
# data_max_angle = 3.05 #175deg


# action_space_lim = int((np.pi * len(bins)) / (2*(data_max_angle - data_min_angle)))
# action = np.arange(-action_space_lim, action_space_lim, 1)
# print(action)
# print(len(action))



def digitize_indexes(labels, bins):
    digit = np.digitize(labels, bins).reshape(-1)
    sort_idx = np.argsort(digit)
    a_sorted = digit[sort_idx]
    unq_first = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))
    unq_items = a_sorted[unq_first]
    unq_count = np.diff(np.nonzero(unq_first)[0])
    return np.split(sort_idx, np.cumsum(unq_count))

#print(_digitize_indexes(labels_training, bins)[4])
#print(np.random.choice(_digitize_indexes(labels_training, bins)[4],1))


import matplotlib.pyplot as plt

class DiscretizedStateSpace(object):

    def __init__(self, step=0.02, is_training=True):
        super(DiscretizedStateSpace, self).__init__()
        self.step = step # Discretization step in radians
        self.is_training = is_training
        self.images, self.labels = None, None
        self.wheel_data, self.wheel_goal =  None, None
        self.action_space_lim = None
        self.cursor = 0
        self._load_dataset()

    def _load_dataset(self):
        if self.is_training:
            print("# Loading training set.")
            self.images = np.load('./data/1dof/usdqn-images-training.npy')[:, 2:-2, 2:-2]
            self.labels = np.load('./data/1dof/usdqn-labels-training.npy')
        else:
            print("# Loading testing set.")
            self.images = np.load('./data/1dof/usdqn-images-testing.npy')[:, 2:-2, 2:-2]
            self.labels = np.load('./data/1dof/usdqn-labels-testing.npy')

        # Data specific information
        data_min_angle = -3.05 #-175deg
        data_max_angle = 3.051 #175deg
        data_goal = [int((np.pi / 2) / self.step), int((np.pi / 2 + np.pi) / self.step)]

        n, bins = np.histogram(self.labels, 
            bins=np.arange(data_min_angle, data_max_angle, self.step))
        print(n)

        self.action_space_lim = int((np.pi * len(bins)) / (2*(data_max_angle - data_min_angle)))

        self.wheel_data = np.array(digitize_indexes(self.labels, bins))

        self.wheel_goal = np.zeros([len(self.wheel_data)])
        self.wheel_goal[data_goal] = 1

        #self.wheel = np.stack([wheel_indexes, wheel_goal], axis=1)

    def reset_wheel(self):
        # Reset state space at random angle
        rnd_ind = np.random.randint(0, len(self.wheel_data))
        self.wheel_data = np.roll(self.wheel_data, rnd_ind, axis=0)
        self.wheel_goal = np.roll(self.wheel_goal, rnd_ind, axis=0)

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

        # Sample random obs from current wheel box
        obs_ind = np.random.choice(self.wheel_data[dangle], 1)[0]

        return (self.images[obs_ind, :, :], 
            self.wheel_goal[dangle], self.get_dist_to_goal(dangle))

    #def get_observation(self):

# class OneDoFRegressionUnsupervised(DiscretizedStateSpace):
#   """docstring for ClassName"""
#   def __init__(self, step=0.02, is_training=True):
#       super(OneDoFRegressionUnsupervised, self).__init__(step=0.02, is_training=True)
#       self.arg = arg
        



state_space = DiscretizedStateSpace(0.1, False)

# print(state_space.wheel)
#state_space.reset()
#state_space.spin(-2)
r = state_space.rotate_wheel(-38)
print("Done:", r[1])
print("Dist:", r[2])
plt.imshow(r[0])
plt.show()
#state_space.spin(+1, keep=True)
#print(np.where(state_space.wheel == 505))


# class BaseDiscretizedStateSpace(object):

#   def __init__(self, arg):
#       super(BaseDiscretizedStateSpace, self).__init__()
#       self.arg = arg
        


# for i in range(images.shape[0]):
#   print("%.02f" % (labels[i] * RAD2DEG))
#   norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(images[i,:,:]), clip=True)
#   mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
#   #scipy.misc.toimage(np.uint8(cm.viridis(images[i,:,:] / 255)*255), cmin=0, cmax=np.max(images[i,:,:])).save('outfile.png')
#   im = Image.fromarray(np.uint8(mapper.to_rgba(images[i,:,:])*255))
#   im.save("im_%.02f.png" % (labels[i] * RAD2DEG))



# imgplot = plt.imshow(images[0, :,:])
# print(np.max(images[0,:,:]))
# imgplot = plt.imshow(np.uint8(cm.viridis(images[0,:,:] / 255)*255), cmin=0, cmax=np.max(images[0,:,:]))
# plt.show()

# new_images = do_obs_processing(images[1,:,:], 84, 84).reshape([1, 84, 84])
# for k in range(1, images.shape[0]):
#     r_img = do_obs_processing(images[k,:,:], 84, 84)
#     r_img = r_img.reshape([1, 84, 84])
#     new_images = np.concatenate([new_images, r_img], axis=0)

# print(new_images.shape)

# np.save('data/1dof/raw/usdqn-images-84x84.npy', new_images)
