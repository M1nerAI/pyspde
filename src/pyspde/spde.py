
from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse
from sksparse import cholmod

from .neighbours import Neighbours
from .utils import map_1d_2d_nx, map_2d_1d_nx_ny, append_element

if TYPE_CHECKING:
    from .grid import Grid

__all__ = ['Spde']

class Spde:
    """Represents a Stochastic Partial Differential Equation (SPDE) solver.

    Attributes
    ----------
    _v : int
        Order of the modified Bessel function of the second kind.
    _d : int
        Dimensions.

    Methods
    -------
    __init__(grid: Grid, sigma: float = 1, a: float = 50,
    dtype: type = np.float64) -> None
        Initialize the SPDE.
    revert_padding(Z_M: np.ndarray) -> np.ndarray
        Reverts the padding.
    create_system() -> sparse.csc_matrix
        Create the system.
    calc_element_terms(A: sparse.dok_matrix, i: int, j: int,
     neigh: Neighbours) -> None
        Calculate the elements of the matrix A.
    calc_derivatives(i: int, j: int, neigh: Neighbours) -> tuple
        Calculate the derivatives at a given point on the grid.
    simulate(n: int = 1, seed: int | None = None) -> np.ndarray
        Simulate the Stochastic Partial Differential Equation solver.

    """

    _v = 1
    _d = 2

    def __init__(self, grid: Grid, sigma: float = 1, a: float = 50,
                 dtype: type = np.float64) -> None:
        """Initialize the SPDE.

        Args:
        ----
            grid (Grid): The grid object to initialize the class.
            sigma (float): The sigma value for calculations. Default is 1.
            a (float): The a value for calculations. Default is 50.
            dtype (type): The data type for calculations. Default is np.float64.

        """
        self.grid = grid
        self.dtype = dtype

        alpha = self._v + self._d / 2

        self.k = 1 / a
        self.tau = ((sigma * (math.gamma(alpha) ** 0.5) *
                    ((4 * np.pi) ** (self._d / 4)) * (self.k**self._v)) /
                    math.gamma(self._v) ** 0.5)

        self.A, self.det_H = self.create_system()



    def revert_padding(self, Z_M: np.ndarray) -> np.ndarray:
        """Reverts the padding.

        Args:
        ----
            Z_M (np.ndarray): The input numpy array.

        Returns:
        -------
            np.ndarray: The numpy array with padding reverted.

        """
        if Z_M.ndim == 2:
            Z_M = Z_M[self.grid.pady:-self.grid.pady,
                      self.grid.padx:-self.grid.padx]
        else:
            Z_M = Z_M[self.grid.pady:-self.grid.pady,
                      self.grid.padx:-self.grid.padx,
                      :]
        return Z_M


    def create_system(self) -> sparse.csc_matrix:
        """Create the system.

        Returns
        -------
            sparse.csc_matrix: The sparse matrix representing the system.

        """
        map_2d_1d = partial(map_2d_1d_nx_ny, nx=self.grid._nx,
                                             ny=self.grid._ny)
        n = self.grid._nx*self.grid._ny

        vals, rows, cols = [], [], []
        det_H = np.ndarray(n, dtype=self.dtype)

        for i in range(self.grid._ny):
            for j in range(self.grid._nx):

                neigh = Neighbours(i, j, map_2d_1d)
                self.calc_element_terms(vals, rows, cols, det_H, i, j, neigh)

        A = sparse.csc_matrix((vals, (rows, cols)))

        return A, det_H

    def calc_element_terms(self, vals: list, rows: list, cols: list,
                           det_H: np.ndarray, 
                           i: int, j: int, neigh: Neighbours) -> None:
        """Calculate the elements of the matrix A.

        Args:
        ----
        A : sparse.dok_matrix
            The matrix to store the calculated elements.
        i : int
            The row index in the matrix.
        j : int
            The column index in the matrix.
        neigh : Neighbours
            Object containing information about neighboring elements in the
            grid.

        """
        a, b, c, d, e, p, q = self.calc_derivatives(i, j, neigh)

        append_element(vals, rows, cols, (2*q + 2*p) + self.k**2,
                       neigh.id, neigh.id)

        H_x = self.grid.anisotropy.H[i, j]
        det_H[neigh.id] = H_x[0,0]*H_x[1,1] - H_x[0,1]*H_x[1,0]

        if neigh.id_bot is not None:
            append_element(vals, rows, cols,- c - p - b,
                           neigh.id, neigh.id_bot)

        if neigh.id_bot_left is not None:
            append_element(vals, rows, cols,e,
                           neigh.id, neigh.id_bot_left)

        if neigh.id_left is not None:
            append_element(vals, rows, cols,- q + d + a,
                           neigh.id, neigh.id_left)

        if neigh.id_top_left is not None:
            append_element(vals, rows, cols, - e,
                            neigh.id, neigh.id_top_left)

        if neigh.id_top is not None:
            append_element(vals, rows, cols,- p + c + b,
                            neigh.id, neigh.id_top)

        if neigh.id_top_right is not None:
            append_element(vals, rows, cols,e,
                             neigh.id, neigh.id_top_right)

        if neigh.id_right is not None:
            append_element(vals, rows, cols,- d - q - a,
                            neigh.id, neigh.id_right)

        if neigh.id_bot_right is not None:
            append_element(vals, rows, cols,- e,
                           neigh.id, neigh.id_bot_right)


    def calc_derivatives(self, i: int, j: int, neigh: Neighbours) -> tuple:
        """Calculate the derivatives at a given point on the grid.

        Parameters
        ----------
        i : int
            The x-coordinate of the point.
        j : int
            The y-coordinate of the point.
        neigh : Neighbours
            Neighbouring points information.

        Returns
        -------
        tuple
            Tuple containing the calculated derivatives a, b, c, d, e, p, q.

        """
        H = self.grid.anisotropy.H
        dx = self.grid.dx
        dy = self.grid.dy

        if neigh.has('id_top') & neigh.has('id_bot'):
            a = (H[i + 1, j, 0, 1] - H[i - 1, j, 0, 1])/(4*dx*dy)
            c = (H[i  + 1, j, 0, 0] - H[i - 1, j, 0, 0])/(4*dy**2)
        else:
            a = 0.0
            c = 0.0

        if neigh.has('id_left') & neigh.has('id_right'):
            b = (H[i, j + 1, 1, 0] - H[i, j - 1, 1, 0])/(4*dx*dy)
            d = (H[i, j + 1, 1, 1] - H[i, j - 1, 1, 1])/(4*dx**2)
        else:
            b = 0.0
            d = 0.0

        e = (H[i, j, 0, 1] + H[i, j, 1, 0])/(4*dx*dy)

        p = H[i, j, 0, 0]/dy**2
        q = H[i, j, 1, 1]/dx**2

        return a, b, c, d, e, p, q


    def simulate(self, n: int = 1, seed: int | None = None) -> np.ndarray:
        """Simulate.

        Parameters
        ----------
        n : int, optional
            Number of simulations to generate. Default is 1.
        seed : int or None, optional
            Random seed for reproducibility. Default is None.

        Returns
        -------
        np.ndarray
            Simulation results in the shape (ny, nx, n).

        """
        rng = np.random.default_rng(seed=seed)

        Q_yy = (self.A.T @ self.A).tocsc() 

        try:
            factor = cholmod.cholesky(Q_yy)

        except cholmod.CholmodNotPositiveDefiniteError:
            msg = ('Anisotropy field is not positive definite, '
                   'check anisotropy.check_positiveness()')
            raise RuntimeError(msg)

        W = rng.standard_normal(size=(Q_yy.shape[0], n))

        Z = factor.solve_Lt(
                W
                *(self.tau**2 * (self.det_H**0.5)[factor.P() ,np.newaxis])**0.5
                ,
                use_LDLt_decomposition=False)

        idxs = np.argsort(factor.P())

        Z = Z[idxs, :]

        Z_M = Z.reshape((self.grid._ny, self.grid._nx, n))

        Z_M = self.revert_padding(Z_M)

        return Z_M


    def kriging(self, samples: np.ndarray) -> np.ndarray:
        """Solve for Simple kriging, based on the provided samples.

        Parameters
        ----------
        samples : np.ndarray
            numpy array of shape (n_samples, 3) containing the sample points
            with x, y coordinates and values.

        Returns
        -------
        np.ndarray
            numpy array of shape (ny, nx) representing the kriged 2D grid.

        """
        xv = samples[:,0].astype(int)
        yv = samples[:,1].astype(int)
        z_ = samples[:,2:]

        if z_.ndim == 1:
            z_ = z_[:, np.newaxis]

        n_samps = z_.shape[0]
        n_vars = z_.shape[1]

        xv += self.grid.padx
        yv += self.grid.pady

        map_2d_1d = partial(map_2d_1d_nx_ny, nx=self.grid._nx,
                                             ny=self.grid._ny)
        map_1d_2d = partial(map_1d_2d_nx, nx=self.grid._nx)

        samples_idxs = []
        for i, j in zip(yv, xv):
            idx = map_2d_1d(i, j)
            samples_idxs.append(idx)


        target_idxs, i_grid, j_grid = [], [], []

        for idx in range(self.grid._nx * self.grid._ny):
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

        Q = (A.T @ A).tocsc()

        Q_yy = Q[n_samps:, n_samps:]
        Q_yx = Q[n_samps:, :n_samps]

        try:
            factor = cholmod.cholesky(Q_yy)

        except cholmod.CholmodNotPositiveDefiniteError:
            msg = ('Anisotropy field is not positive definite, '
                   'check anisotropy.check_positiveness()')
            raise RuntimeError(msg)


        Z = -factor.solve_A(Q_yx @ z_)

        Z_M = np.empty((self.grid._ny, self.grid._nx, n_vars))
        Z_M[i_grid, j_grid, :] = Z
        Z_M[yv, xv, :] = z_

        Z_M = self.revert_padding(Z_M)\
                  .squeeze()

        return Z_M


