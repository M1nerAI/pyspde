import pyspde
import matplotlib.pyplot as plt

# Create Anisotropy
atpy = pyspde.anisotropy_from_svg(r'assets/anticline.svg', beta=50, gamma=0.5)
atpy.plot()

# Create Grid
grid = pyspde.Grid(nx=200, ny=100, anisotropy=atpy)

# Create SPDE
sp = pyspde.Spde(grid, sigma=1, a=10)

# Simulate
Z = sp.simulate(seed=0)

# Sample
X = pyspde.sample_grid(Z[:,:,0], 20, 10)

# Kriging
Y = sp.kriging(X)

# Plot
plt.imshow(Y)
plt.show()
