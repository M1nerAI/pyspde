import pyspde
import matplotlib.pyplot as plt

# Create Anisotropy
atpy = pyspde.anisotropy_from_svg(r'assets/anticline.svg', beta=50, gamma=0.5)

# Create Grid
grid = pyspde.Grid(nx=800, ny=400, anisotropy=atpy, padx=100, pady=100)

# Create SPDE
sp = pyspde.Spde(grid, sigma=1, a=50)

# Simulate
Z = sp.simulate(seed=0)

# Plot
plt.imshow(Z[:,:,0])
plt.show()
