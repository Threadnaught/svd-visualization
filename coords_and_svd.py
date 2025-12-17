import numpy as np

np.random.seed(5)

n=200
dims=[100,25,50]

coords_world = np.transpose([np.random.normal(scale=x,size=n) for x in dims])

# Compute the SVD:
U, S, Vh = np.linalg.svd(coords_world)

# Pre-align our coordinate system with the computed SVD.
# This is a little cheaty but since we're going to be rotating them all over the place it doesn't REALLY matter
coords_world = np.matmul(coords_world, Vh)
