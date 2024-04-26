import numpy as np
from src.pyspde import anisotropy_from_svg, anisotropy_stationary, Grid, Spde
import pickle
import matplotlib.pyplot as plt

from src.pyspde.utils import sample_grid
        
#@profile
def main():
    nx = 400
    ny = 200
    seed = None
    beta = 100
    gamma = 1
    a = 15

    # sampling
    dx = 15
    dy = 25

    #atrp = anisotropy_stationary(theta=theta, gamma=gamma, beta=beta)
    atrp = anisotropy_from_svg(r'assets/galaxy.svg', beta=beta, gamma=gamma, k=10, norm_type='max')
    #atrp = anisotropy_from_svg(r'assets/anticline.svg', beta=beta, gamma=gamma, k=3, norm_type='norm')

    grid = Grid(nx=nx, ny=ny, anisotropy=atrp, padx=2*a, pady=2*a)
    sp = Spde(grid, sigma=1, a=a)

    Z = sp.simulate(seed=seed)


    X = sample_grid(Z, dx, dy)


    Y = sp.kriging(X)


    fig, ax = plt.subplots(1, 2)


    ax[0].imshow(Z,
            vmin=X[:,2].min(), vmax=X[:,2].max(),)
    ax[0].set_title('Non Conditional Simulation')
    ax[1].imshow(Y,
            vmin=X[:,2].min(), vmax=X[:,2].max())
    ax[1].set_title('Samples and Kriging')
    ax[1].scatter(X[:,0], X[:,1], c=X[:,2],
                 vmin=X[:,2].min(), vmax=X[:,2].max(),
                 linewidths=0.5, 
                 edgecolors='black')

    fig.suptitle('Messier 81 Anisotropy')
    plt.tight_layout()
    fig.show()
    breakpoint()

if __name__ == '__main__':
    main()