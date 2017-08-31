import numpy as np
import matplotlib
import matplotlib.cm as cm
from PIL import Image

RAD2DEG = 57.29577951308232

labelstrain = np.load('./data/1dof/usdqn-labels-training.npy')
labelstest = np.load('./data/1dof/usdqn-labels-testing.npy')

imagestrain = np.load('./data/1dof/usdqn-images-training.npy')
imagestest = np.load('./data/1dof/usdqn-images-testing.npy')


# labels = labelstrain.reshape(-1)
# images = imagestrain

labels = labelstest.reshape(-1)
images = imagestest

print(labels.shape)
print(images.shape)

data_lim = (3.05-np.pi)

#print(labels[np.where((labels > data_lim) &  (labels < -data_lim))] + np.pi)

second = np.flip(np.where((labels > data_lim) &  (labels < 0))[0], axis=0)
second = np.where((labels > data_lim) &  (labels < 0))
second_labels = labels[second] + np.pi
second_images = np.flip(np.fliplr(images[second,:,:]).reshape([-1, 84,84]), axis=0)
print(labels[second] + np.pi)


first = np.flip(np.where((labels > 0) &  (labels < -data_lim))[0], axis=0)
first = np.where((labels > 0) &  (labels < -data_lim))
first_labels = labels[first] - np.pi
first_images = np.fliplr(images[first,:,:]).reshape([-1, 84,84])
print(labels[first] - np.pi)
print(labels[first])


print(second_images.shape)
print(first_images.shape)

labels = np.concatenate([first_labels, labels, second_labels], axis=0)
images = np.concatenate([first_images, images, second_images], axis=0)
print(labels)
print(images.shape)


# np.save('./data/1dof/usdqn-labels-training-v2.npy', labels)
# np.save('./data/1dof/usdqn-images-training-v2.npy', images)

np.save('./data/1dof/usdqn-labels-testing-v2.npy', labels)
np.save('./data/1dof/usdqn-images-testing-v2.npy', images)



# for i in range(images.shape[0]):
# 	print("%.02f" % (labels[i] * RAD2DEG))
# 	norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(images[i,:,:]), clip=True)
# 	mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
# 	#scipy.misc.toimage(np.uint8(cm.viridis(images[i,:,:] / 255)*255), cmin=0, cmax=np.max(images[i,:,:])).save('outfile.png')
# 	im = Image.fromarray(np.uint8(mapper.to_rgba(images[i,:,:])*255))
# 	im.save("out/im_%.02f.png" % (i))