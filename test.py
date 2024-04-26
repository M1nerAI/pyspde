from src.pyspde import anisotropy, Grid, Spde, anisotropy_from_svg
import matplotlib.pyplot as plt



atpy = anisotropy_from_svg(r'assets/anticline.svg', beta=50, gamma=0.5)
grid = Grid(nx=400, ny=200, anisotropy=atpy)

sp = Spde(grid, sigma=1, a=50)
Z = sp.simulate(seed=0)
plt.imshow(Z)
plt.tight_layout()

plt.savefig('anticline_simu.png', dpi=600, bbox_inches='tight')