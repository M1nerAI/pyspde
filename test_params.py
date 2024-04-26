import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from src.pyspde import anisotropy_from_svg, anisotropy_stationary, Grid, Spde

import traceback



seed = 1613001354
theta = 45
#gammas = [0.01, 0.05, 0.1, 0.5, 1, 5, 10]
#betas = [0.01, 0.05, 0.1, 0.5, 1, 5, 10]
gammas = [1,2,10,20]
betas = [100,150,300,400]

fig, ax = plt.subplots(len(gammas),len(betas))

for i, gamma in enumerate(gammas):
    for j, beta in enumerate(betas):

        #atrp = anisotropy_stationary(theta=theta, gamma=gamma, beta=beta)
        #atrp = anisotropy_from_svg(r'assets/galaxy.svg', beta=beta, gamma=gamma, k=3, norm_type='max')
        atrp = anisotropy_from_svg(r'assets/anticline.svg', beta=beta, gamma=gamma, k=3, norm_type='norm')


        #atrp.plot()

        grid = Grid(nx=200, ny=100, anisotropy=atrp, padx=10, pady=10)

        #atrp.plot()
        #breakpoint()

        sp = Spde(grid, sigma=1, a=3)
        try:
            Z = sp.simulate(seed=seed)

        except  Exception as e:
            print(traceback.format_exc())
            print(type(e))

            breakpoint()
            Po = atrp.check_positiveness()
            Diff = atrp.check_differenciable(1 , 1)

            plt.close('all')
            fig, ax = plt.subplots(2, 2)

            ax[0,0].imshow(Diff[:,:,0,0])
            ax[0,1].imshow(Diff[:,:,0,1])
            ax[1,0].imshow(Diff[:,:,1,0])
            ax[1,1].imshow(Diff[:,:,1,1])

            fig.show()


        ax[i, j].imshow(Z, vmax=2, vmin=-2)
        ax[i, j].set_title(f'mean {Z.mean():.3f} std {Z.std():.3f}')

        if j == 0:
            ax[i, j].set_ylabel(f'gamma = {gamma}')

        if i == 0:
            ax[i, j].set_xlabel(f'beta = {beta}')
            ax[i, j].xaxis.set_label_position('top')


        ax[i, j].get_xaxis().set_ticks([])
        ax[i, j].get_yaxis().set_ticks([])

plt.tight_layout()

fig.show()
breakpoint()