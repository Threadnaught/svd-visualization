import numpy as np
import matplotlib.pyplot as plt
from coords_and_svd import n, coords_world, S, Vh

plot_hist = True
large_dim_dataset = True

if large_dim_dataset:
    coords_world = np.random.normal(size=[1000, 1000])
    U, S, Vh = np.linalg.svd(coords_world)
    print(S.shape)

if not large_dim_dataset:
    plt.plot([0,S[0]], [0.4,0.4], 'r')
    plt.plot([0,S[1]], [0.5,0.5], 'g')
    plt.plot([0,S[2]], [0.6,0.6], 'b')
    plt.ylim(0,1.5)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('Singular value')

if plot_hist:
    plt.hist(S, bins=20, color='black')
    plt.ylabel('Count')
else:    
    ax.spines['left'].set_visible(False)
    plt.yticks([])

plt.savefig('imgs/esd-hist-%r-large-%r.png' % (plot_hist, large_dim_dataset))
plt.show()
