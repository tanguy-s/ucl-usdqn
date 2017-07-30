import gym
import numpy as np
# env = gym.make('CartPole-v0')
# env.reset()

# print(env.action_space.sample())
# for _ in range(1000):
#     print(env.action_space.sample())
#     env.render()
#     env.step(env.action_space.sample()) # take a random action


import matplotlib.pyplot as plt



labels_orig = np.load('data/1dof/raw/usdqn-labels.npy')

print(labels_orig)

print(labels_orig.shape)

im_px = labels_orig.shape[0] / 350
 
print(im_px)

labels, unique_idx = np.unique(np.round(labels_orig, 4), return_index=True)
print(labels)
print(unique_idx)

n = 0
j = 0
step = 0.005
steps = np.arange(-3.05, 3.06, step)
smpl_count = np.ones(steps.shape)
smpls_tokeep = list()
for i in range(0, steps.shape[0]):
    smpl_count[i] = np.bitwise_and((labels >= steps[i]), (labels < steps[i] + step)).sum()

    smpls_indx = np.where((labels >= steps[i]) & (labels < steps[i] + step))
    if len(smpls_indx) > 0 and len(smpls_indx[0]) > 0:
        smpls = labels[smpls_indx[0][0]]
        smpls_tokeep.append(smpls_indx[0][0])
        j += 1
    else:
        smpls = []
    print(smpls)

dataset_idx = unique_idx[smpls_tokeep]

testing_idx = np.arange(0, len(dataset_idx), 3)

labels_testing_idx = dataset_idx[testing_idx]
labels_training_idx = np.delete(dataset_idx, testing_idx)

print("Dataset total len: %s" % len(dataset_idx))
print("Dataset training len: %s" % len(labels_training_idx))
print("Dataset testing len: %s" % len(labels_testing_idx))
print("Training + Testing len: %s" % (len(labels_training_idx) + len(labels_testing_idx)))

images = np.load('data/1dof/raw/usdqn-images-84x84.npy')

training_images = images[labels_training_idx, :, :]
testing_images = images[labels_testing_idx, :, :]

training_labels = labels_orig[labels_training_idx]
testing_labels = labels_orig[labels_testing_idx]

np.save('data/1dof/usdqn-labels-training.npy', training_labels)
np.save('data/1dof/usdqn-labels-testing.npy', testing_labels)
np.save('data/1dof/usdqn-images-training.npy', training_images)
np.save('data/1dof/usdqn-images-testing.npy', testing_images)


# print(smpl_count.shape)
# print("Samples count mean: %s std: %s" % (np.mean(smpl_count), np.std(smpl_count)))
# print(np.arange(-3.05, 3.06, step).shape)
# plt.bar(np.arange(-3.05, 3.06, step), smpl_count)

# plt.grid(True)
# #plt.savefig("test.png")
# plt.show()

