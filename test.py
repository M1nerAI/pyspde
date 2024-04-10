import matplotlib.pyplot as plt

from src import anisotropy
from src import grid
from src import spde


sigma = 1
gamma = 0.1
beta = 10
a = 25

nx = 200
ny = 100

anis = anisotropy.from_svg(r'C:\Users\jimen\Documents\00.Repos\Minerai\studies\anis.svg', beta, gamma)

gd = grid.Grid(nx=nx, ny=ny, dx=1, dy=1, anisotropy=anis)

sp = spde.Spde(gd, sigma, a)
Z_M = sp.simulate(n=5)

fig, ax = plt.subplots()

ax.imshow(Z_M[:,:,2])

fig.show()
breakpoint()