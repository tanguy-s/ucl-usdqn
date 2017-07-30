import warnings
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt

def do_obs_processing(frame, width, height):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return resize(frame, (width, height))

images = np.load('data/1dof/raw/usdqn-images-1.npy')
# for k in range(2,9):
#     print(k)
#     imgs = np.load('data/1dof/raw/usdqn-images-%s.npy' % k)
#     images = np.concatenate([images, imgs], axis=0)

print(images.shape)

images = images.reshape(-1, 616, 820)


imgplot = plt.imshow(images[0, :,:])
plt.show()

# new_images = do_obs_processing(images[1,:,:], 84, 84).reshape([1, 84, 84])
# for k in range(1, images.shape[0]):
#     r_img = do_obs_processing(images[k,:,:], 84, 84)
#     r_img = r_img.reshape([1, 84, 84])
#     new_images = np.concatenate([new_images, r_img], axis=0)

# print(new_images.shape)

# np.save('data/1dof/raw/usdqn-images-84x84.npy', new_images)
