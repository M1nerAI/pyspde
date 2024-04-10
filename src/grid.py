
from typing import Union
import numpy as np
from .anisotropy import Anisotropy
from  scipy.spatial import  KDTree

class Grid:
    _padx_pct = 15
    _pady_pct = 15

    def __init__(self, nx: int, ny: int, dx: float, dy: float,
                 anisotropy: Anisotropy, padx: None | int = None,
                 pady: None | int = None):

        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.anisotropy = anisotropy

        # Add Padding
        if padx is None:
            self.padx = int(np.ceil(self._padx_pct / 100 * nx))

        if pady is None:
            self.pady = int(np.ceil(self._pady_pct / 100 * ny))

        self._nx = nx + 2 * self.padx
        self._ny = ny + 2 * self.pady

        self.intrp_anisotropy()


    def intrp_anisotropy(self, k: int = 3, scale: bool = True):

        anis = self.anisotropy

        if scale:
            scale_x = self.nx * self.dx / anis.width
            scale_y = self.ny * self.dy / anis.height

            # TODO scale u,v 
        else:
            scale_x = 1
            scale_y = 1

        x = (anis.x * scale_x) + self.padx
        y = (anis.y * scale_y) + self.pady

        X = np.column_stack((x, y))
        U = np.column_stack((anis.u, anis.v))

        # 1D
        grid = np.meshgrid(np.arange(self._nx) * self.dx,
                           np.arange(self._ny) * self.dy)

        # 2D
        grid = np.column_stack((grid[0].reshape(-1), grid[1].reshape(-1)))

        V = np.zeros(grid.shape, dtype=anis.dtype)
        tree = KDTree(X)

        dists, idxs = tree.query(grid, k=k)

        for i,(dist, idx) in enumerate(zip(dists, idxs)):

            inv_dist = dist**(-1)
            inv_dist = inv_dist/inv_dist.sum()
            V[i, :] = (U[idx, :] * inv_dist[:, np.newaxis]).sum(axis=0)

        V = V.reshape(self._ny, self._nx , 2)[:,:,::-1]
        H = np.zeros((self._ny,self._nx, 2, 2), dtype=anis.dtype)
        I = np.eye(2, dtype=anis.dtype)

        for i in range(self._ny):
            for j in range(self._nx):
                H[i, j, :, :] = anis.gamma*I +\
                                anis.beta*(V[i, j][:,np.newaxis] @ \
                                           V[i, j][np.newaxis,:] )

        anis.H = H
        anis.V = V
