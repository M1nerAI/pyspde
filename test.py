import matplotlib.pyplot as plt

from src.pyspde import anisotropy_from_svg, Grid, Spde

atrp = anisotropy_from_svg(r'assets/anticline.svg', beta=2, gamma=0.005)
#atrp.plot(show=True)
grid = Grid(nx=200, ny=100, anisotropy=atrp)
sp = Spde(grid, sigma=1, a=50)
Z = sp.simulate(seed=10000)

fig, ax = plt.subplots()
ax.imshow(Z)

fig.show()
breakpoint()