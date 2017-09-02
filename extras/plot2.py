import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

RAD2DEG = 57.29577951308232


testing = np.genfromtxt('./usdqn_tf/dumps/1dof_da_u_4/1/testing.csv', delimiter=',')

bins = np.arange(-np.pi, np.pi, 0.04)


# theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
#radii = max_height*np.random.rand(N)
#print(radii)
width = 2 * np.pi / 720


ax = plt.subplot(111, polar=True) #5 #8 #11
ax.set_theta_zero_location("N")

testing[:,2] = testing[:,2] / np.max(testing[:,2])
print(testing[:,11])
bars = ax.bar(bins[:-1], testing[:,2], bottom=1, width=0.04, label='Raw img count')
#bars = ax.plot(bins[:-1], testing[:,8], label='Raw img count')
#bars = ax.plot(bins[:-1], testing[:,8], label='Raw img count')

#ax.set_ybound(lower=-3, upper=1)



# # Use custom colors and opacity
max_val = np.max(testing[:,2])
testing[:,2] = 4 * ((testing[:,2] - 0.5) / max_val)
for r, bar in zip(testing[:,2], bars):
    bar.set_facecolor(plt.cm.jet(r / 10.))
    bar.set_alpha(0.8)

plt.show()