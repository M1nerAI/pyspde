import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from src.pyspde import anisotropy_from_svg, anisotropy_stationary, Grid, Spde

#atrp = anisotropy_from_svg(r'assets/anticline.svg', beta=2, gamma=0.005)
nSeconds = 0.2
fps = 3

thetas = np.arange(0,180,5)



thetas = np.arange(0,180,5)
gammas = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50]
betas = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50]

Zs = []





Zs = []
for theta in thetas:

    theta = 45
    atrp = anisotropy_stationary(theta=theta, gamma=1, beta=8)
    grid = Grid(nx=200, ny=100, anisotropy=atrp, padx=10, pady=10)
    sp = Spde(grid, sigma=1, a=1)

    Zs.append(sp.simulate(
        seed=10000
        ))

    print('theta', theta, 'mean', np.mean(Zs[-1]), 'stev', np.std(Zs[-1]))

fig, ax = plt.subplots()
im = ax.imshow(Zs[0], vmax=3, vmin=-3)

def animate_func(i):
    im.set_array(Zs[i])
    return [im]

anim = animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               frames = len(Zs),
                               repeat = False,
                               #interval = 1000 / fps, # in ms
                               )

anim.save('test_anim.gif', fps=fps)
