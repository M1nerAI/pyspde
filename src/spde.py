
import math

from functools import partial


import numpy as np
from scipy import sparse
from sksparse import cholmod


from .grid import Grid
from .neighbours import Neighbours
from .utils import map_1d_2d_nx, map_2d_1d_nx_ny


class Spde:
    _v = 1
    _d = 2

    def __init__(
        self,
        grid: Grid,
        sigma: float = 1,
        a: float = 50,
        dtype: type = np.float64
    ):
        self.grid = grid
        self.dtype = dtype

        alpha = self._v + self._d / 2

        self.k = 1 / a
        self.tau = ((sigma * (math.gamma(alpha) ** 0.5) *
                    ((4 * np.pi) ** (self._d / 4)) * (self.k**self._v)) /
                    math.gamma(self._v) ** 0.5)
        self.A = self.create_system()


    def revert_padding(self, Z_M):

        if Z_M.ndim == 2:
            Z_M = Z_M[self.grid.pady:-self.grid.pady,
                      self.grid.padx:-self.grid.padx]
        else:
            Z_M = Z_M[self.grid.pady:-self.grid.pady,
                      self.grid.padx:-self.grid.padx,
                      :]
        return Z_M
    

    def create_system(self):
        
        map_2d_1d = partial(map_2d_1d_nx_ny, nx=self.grid._nx,
                                             ny=self.grid._ny)
        n = self.grid._nx*self.grid._ny

        A = sparse.dok_array((n, n), dtype=self.dtype)

        for i in range(self.grid._ny):
            for j in range(self.grid._nx):

                neigh = Neighbours(i, j, map_2d_1d)
                self.calc_element_terms(A, i, j, neigh)

        A = sparse.csc_matrix(A)

        return A 


    def calc_element_terms(self, A, i, j, neigh):

        a, b, c, d, e, p, q = self.calc_derivatives(i, j, neigh)
            
        A[neigh.id, neigh.id] = (2*q + 2*p) + self.k**2

        if neigh.id_bot is not None:
            A[neigh.id, neigh.id_bot] = - c - p - b

        if neigh.id_bot_left is not None:
            A[neigh.id, neigh.id_bot_left] = e

        if neigh.id_left is not None:
            A[neigh.id, neigh.id_left] = - q + d + a

        if neigh.id_top_left is not None:
            A[neigh.id, neigh.id_top_left] = - e

        if neigh.id_top is not None:
            A[neigh.id, neigh.id_top] = - p + c + b

        if neigh.id_top_right is not None:
            A[neigh.id, neigh.id_top_right] = e

        if neigh.id_right is not None:
            A[neigh.id, neigh.id_right] = - d - q - a

        if neigh.id_bot_right is not None:
            A[neigh.id, neigh.id_bot_right] = - e


    def calc_derivatives(self, i, j, neigh):

        H = self.grid.anisotropy.H
        dx = self.grid.dx
        dy = self.grid.dy

        if ['id_top', 'id_bot'] in neigh.allowable_neighs:
            a = (H[i + 1, j, 0, 1] - H[i - 1, j, 0, 1])/(4*dx*dy)
            c = (H[i  + 1, j, 0, 0] - H[i - 1, j, 0, 0])/(4*dy**2)
        else:
            a = 0
            c = 0

        if ['id_left', 'id_right'] in neigh.allowable_neighs:
            b = (H[i, j + 1, 1, 0] - H[i, j - 1, 1, 0])/(4*dx*dy)
            d = (H[i, j + 1, 1, 1] - H[i, j - 1, 1, 1])/(4*dx**2)
        else:
            b = 0
            d = 0

        e = (H[i, j, 0, 1] + H[i, j, 1, 0])/(4*dx*dy)

        p = H[i, j, 0, 0]/dy**2
        q = H[i, j, 1, 1]/dx**2

        return a, b, c, d, e, p, q


    def simulate(self, n: int = 1):

        Q_yy = (self.A.T @ self.A) / self.tau**2
        factor = cholmod.cholesky(Q_yy)

        Z = factor.solve_Lt(
                np.random.normal(0, 1, size=(Q_yy.shape[0], n)),
                use_LDLt_decomposition=False) 
        
        idxs = np.argsort(factor.P())

        Z = Z[idxs, :]

        Z_M = Z.reshape((self.grid._ny, self.grid._nx, n))

        Z_M = self.revert_padding(Z_M)

        return Z_M


    def kriging(self, samples):

        xv = samples[:,0]
        yv = samples[:,1]
        z_ = samples[:,3]

        n_samps = z_.shape[0]

        xv += self.grid.padx
        yv += self.grid.pady

        map_2d_1d = partial(map_2d_1d_nx_ny, nx=self._nx, ny=self._ny)
        map_1d_2d = partial(map_1d_2d_nx, nx=self._nx)

        samples_idxs = []
        for i, j in zip(yv, xv):
            idx = map_2d_1d(i, j)
            samples_idxs.append(idx)

        target_idxs, i_grid, j_grid = [], [], []

        for idx in range(self._nx * self._ny):
            if idx not in samples_idxs:
                target_idxs.append(idx)
                i, j = map_1d_2d(idx)
                i_grid.append(i)
                j_grid.append(j)

        A_xx = self.A[samples_idxs, :][:, samples_idxs]
        A_yx = self.A[target_idxs, :][:, samples_idxs]
        A_yy = self.A[target_idxs, :][:, target_idxs]

        A_xy = A_yx.T

        A = sparse.vstack([sparse.hstack([A_xx, A_xy]),
                           sparse.hstack([A_yx, A_yy])])

        Q = (A.T @ A) / self.tau**2

        Q_yy = Q[n_samps:, n_samps:]
        Q_yx = Q[n_samps:, :n_samps]

        factor = cholmod.cholesky(Q_yy)

        Z = -factor.solve_A(Q_yx @ z_[:, np.newaxis])[:, 0]

        Z_M = np.empty((ny, nx))
        Z_M[i_grid, j_grid] = Z
        Z_M[yv, xv] = z_

        Z_M = self.revert_padding(Z_M)

        return Z_M
