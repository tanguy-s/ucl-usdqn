import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

N = 3

plt.style.use('seaborn-paper')
print(plt.style.available)

random_res = [-41.083031, -72.820000, 73.820000];
random_std = [27.647397,76.222619, 76.222619];

notrain_res = [-33.21795854, -48.222, 50.209];
notrain_std = [ 19.67563024,  36.31295916, 36.33234786];

dqn_res = [-1.125158, -1.150000, 3.140000];
dqn_std = [0.462151, 0.476970, 0.510294];


ind = np.arange(N)  # the x locations for the groups
width = 0.2       # the width of the bars

fig, ax = plt.subplots()


random = ax.bar(ind, [random_res[0], notrain_res[0], dqn_res[0]], width, yerr=[random_std[0], notrain_std[0], dqn_std[0]], alpha=0.5, 
        # with color
        error_kw=dict(lw=1, capsize=3, capthick=1))
notrain = ax.bar(ind + width, [random_res[1], notrain_res[1], dqn_res[1]], width, yerr=[random_std[1], notrain_std[1], dqn_std[1]], alpha=0.5, 
        # with color
        error_kw=dict(lw=1, capsize=3, capthick=1))
dqn = ax.bar(ind + width + width, [random_res[2], notrain_res[2], dqn_res[2]], width, yerr=[random_std[2], notrain_std[2], dqn_std[2]], alpha=0.5, 
        # with color
        error_kw=dict(lw=1, capsize=3, capthick=1))


# add some text for labels, title and axes ticks
ax.set_ylabel('Values')
#ax.set_title('Scores by group and gender')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Random policy', 'Untrained DDQN', 'selected DDQN'))

plt.ylim([-90, 90] )

ax.legend((random[0], notrain[0], dqn[0]), ('Return', 'Total reward', 'Episode length'), fontsize=9)


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.text(rect.get_x() + rect.get_width()/2., 1*height +1,
                    '%.02f' % height,
                    ha='center', va='bottom', fontsize=9)
        else:
            ax.text(rect.get_x() + rect.get_width()/2., 1*height - 8,
                    '%.02f' % height,
                    ha='center', va='bottom', fontsize=9)

autolabel(random)
autolabel(notrain)
autolabel(dqn)

plt.grid()
plt.show()