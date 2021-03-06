import numpy as np
import matplotlib.pyplot as plt


RAD2DEG = 57.29577951308232

N = 80
bottom = 0
max_height = 4

Num = 610

labels = np.load('../data_raw/1dof/raw/usdqn-labels.npy')


labelstrain = np.load('./data/1dof/usdqn-labels-training.npy')
print(labelstrain.shape[0])
labelstest = np.load('./data/1dof/usdqn-labels-testing.npy')
print(labelstest.shape[0])
dte = np.diff(labelstest, axis=0)
dtr = np.diff(labelstrain, axis=0)
print(dte)
print("Max training dist:", np.max(dte))
print("Max test dist:", np.max(dte))

Num = labels.shape[0]
print(Num)


n, bins, patches = plt.hist(labelstest, bins=np.arange(-np.pi, np.pi, 0.019))
print(len(bins))
print(n)

print("num:", len(np.arange(-180, 180, 0.5)))

# max_val = np.max(n)
# n = max_height * (n / max_val)
print(n)

# theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
radii = max_height*np.random.rand(N)
print(radii)
width = 2 * np.pi / 720
#n = n + np.random.randint(-1, 3, n.shape)
mn = np.mean(n)
print(mn)
print(np.min(n))
print(np.max(n))
print(np.std(n))

n[0:10] = 0
n[-9:] = 0

n2 = np.zeros_like(n)
n2[0:10] = n[int(len(n)/2):int(len(n)/2)+10]
n2[-9:] = n[int(len(n)/2)-9:int(len(n)/2)]

ax = plt.subplot(111, polar=True)
ax.set_theta_zero_location("N")

bars = ax.bar(bins[:-1], n, width=0.01, bottom=0, label='Raw img count')
bars3 = ax.bar(bins[:-1], n2, width=0.01, bottom=0, label='Completed img count')
bars1 = ax.bar(labelstest, np.ones_like(labelstest), width=width, bottom=-5, label='Test set')
bars2 = ax.bar(labelstrain, np.ones_like(labelstrain), width=width, bottom=-3, label='Training set')
goalline = ax.plot((np.pi / 2, 3*np.pi / 2), ( 13, 13), color='k', zorder = 3, label='Needle position')
#meanline = ax.plot(bins[:-1], np.tile(mn, 610), 'r--', linewidth=.05, label='Mean')
#meanline = plt.Circle((0, 0), mn, color='r', fill=False)
#meanline = plt.Circle((0, 0),2*mn+1.8, transform=ax.transData._b, fill=False, color="red", alpha=0.9 , label='Mean')
#ax.add_artist(meanline)
#plt.gcf().gca().add_artist(meanline)
ax.set_ybound(lower=-13, upper=11)


anyArtist = plt.Line2D((0,1),(0,0), color='r')

handles, labels = ax.get_legend_handles_labels()
display = (0,1,2,3,4)

#Create custom artists
#simArtist = plt.Line2D((0,1),(0,0), color='k', marker='o', linestyle='')
#anyArtist = plt.Line2D((0,1),(0,0), color='r')

#Create legend from custom artist/label lists
ax.legend([handle for i,handle in enumerate(handles) if i in display],
          [label for i,label in enumerate(labels) if i in display], fontsize=8, bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)

#ax.legend([bars, meanline], ['Img count', 'Mean'])
# ax.legend(handles=[bars, meanline], bbox_to_anchor=(1, 1),
#            bbox_transform=plt.gcf().transFigure)


# # Use custom colors and opacity
max_val = np.max(n)
n = max_height * ((n - 0.5) / max_val)
for r, bar in zip(n, bars):
    bar.set_facecolor(plt.cm.jet(r / 10.))
    bar.set_alpha(0.8)

plt.show()