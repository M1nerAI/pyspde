
from __future__ import annotations
from typing import TYPE_CHECKING
import warnings

import numpy as np
from  scipy.spatial import  KDTree

if TYPE_CHECKING:
    from .anisotropy import Anisotropy

__all__ = ['Grid']

class Grid:
    """Grid class representing a computational grid for anisotropic simulations.

    Attributes
    ----------
        nx (int): Number of grid points in the x-direction.
        ny (int): Number of grid points in the y-direction.
        dx (float): Spacing between grid points in the x-direction.
        dy (float): Spacing between grid points in the y-direction.
        anisotropy (Anisotropy): An Anisotropy object representing the
        anisotropy field.
        padx (int): Padding in the x-direction.
        pady (int): Padding in the y-direction.
        _nx (int): Total number of grid points including padding in the
        x-direction.
        _ny (int): Total number of grid points including padding in the
        y-direction.

    """

    _padx_pct = 15
    _pady_pct = 15

    def __init__(self, nx: int, ny: int, anisotropy: Anisotropy, dx: float=1,
                 dy: float=1, padx: None | int = None, pady: None | int = None,
                 ) -> None:
        """Initialize a Grid object.

        Args:
        ----
            nx (int): Number of grid points in the x-direction.
            ny (int): Number of grid points in the y-direction.
            dx (float): Spacing between grid points in the x-direction.
            dy (float): Spacing between grid points in the y-direction.
            anisotropy (Anisotropy): An Anisotropy object representing the
            anisotropy field.
            padx (None | int, optional): Padding in the x-direction. Defaults
            to None.
            pady (None | int, optional): Padding in the y-direction. Defaults
            to None.

        """
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.anisotropy = anisotropy

        # Add Padding
        if padx is None:
            self.padx = int(np.ceil(self._padx_pct / 100 * nx))
        else:
            self.padx = padx

        if pady is None:
            self.pady = int(np.ceil(self._pady_pct / 100 * ny))
        else:
            self.pady = pady

        self._nx = nx + 2 * self.padx
        self._ny = ny + 2 * self.pady

        self.intrp_anisotropy()


    def intrp_anisotropy(self) ->  None:
        """Interpolate the anisotropy field onto the grid.

        Args:
        ----
            k (int, optional): Number of nearest neighbors to consider in
            interpolation. Defaults to 3.


        """
        anis = self.anisotropy
        k = min(anis.k,  len(anis.x))


        if anis.scale:
            scale_x = self.nx * self.dx / anis.width
            scale_y = self.ny * self.dy / anis.height

            # TODO(ejimenez): scale u,v
        else:
            scale_x = 1
            scale_y = 1

        x = (anis.x * scale_x) + self.padx
        y = (anis.y * scale_y) + self.pady

       #import matplotlib.pyplot as plt
       #plt.imshow(np.random.rand(self._ny, self._nx))
       #plt.scatter(x, y, c=anis.u, cmap='jet')

       #plt.show()
       #breakpoint()

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

        if k == 1:
            dists = dists[:, np.newaxis]
            idxs = idxs[:, np.newaxis]

        warnings.filterwarnings("error")

        for i,(dist, idx) in enumerate(zip(dists, idxs)):

            try:
                inv_dist = dist**(-0.25)
                inv_dist = inv_dist/inv_dist.sum()
            except RuntimeWarning:

                inv_dist = np.zeros(k, dtype=anis.dtype)
                inv_dist[np.argmin(dist)] = 1 


            V[i, :] = (U[idx, :] * inv_dist[:, np.newaxis]).sum(axis=0)

        warnings.resetwarnings()

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


