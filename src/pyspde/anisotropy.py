import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import xml.etree.ElementTree as Et
from matplotlib.figure import Figure

from .utils import get_inkscape_namespace, get_inkscape_transform

__all__ = ['Anisotropy', 'anisotropy_from_svg', 'anisotropy_stationary']

class Anisotropy:
    """Represents an anisotropy object.

    Attributes
    ----------
        _x (np.ndarray): X-coordinates of vectors.
        _y (np.ndarray): Y-coordinates of vectors.
        _u (np.ndarray): Horizontal components of vectors.
        _v (np.ndarray): Vertical components of vectors.
        width (float): Width extent of the anisotropy.
        height (float): Height extent of the anisotropy.
        H (np.ndarray): Anisotropy tensor.
        V (np.ndarray): Vector field tensor.
        beta (float): Anisotropic local effect coefficient.
        gamma (float): Isotropic baseline effect coeficient.
        dtype (type): Data type for the fields tensors.

    """

    def __init__(self, x_u: np.ndarray, beta: float, gamma: float,
                 width: float = None, height: float = None, k: int = 3,
                 dtype: type = np.float64, *, scale: bool = True) -> None:
        """Initialize an Anisotropy object.

        Args:
        ----
            x_u (np.ndarray): Anisotropy data points of shape (p, 4). Each row
            is composed by  (x, y, u, v) where (x, y) coordinates with (u, v)
            directions.
            beta (float): Anisotropic local effect coefficient.
            gamma (float): Isotropic baseline effect coeficient.
            width (float): Width extent of the anisotropy.
            height (float): Height extent of the anisotropy.
            k (int): No of neighbours to consider during interpolation.
            dtype (type): Data type for the fields tensors.
            Defaults to np.float32.
            scale (bool, optional): Whether to scale the anisotropy field to
            match grid dimensions. Defaults to True.

        """
        # No Interpolation
        self._x = x_u[:,0].astype(dtype)
        self._y = x_u[:,1].astype(dtype)
        self._u = x_u[:,2].astype(dtype)
        self._v = x_u[:,3].astype(dtype)

        self.width = x_u[:, 0].max() - x_u[:, 0].min() if width is None \
                     else width

        self.height = x_u[:, 1].max() - x_u[:, 1].min() if height is None \
                     else height

        # Interpolated
        self.H = None
        self.V = None

        # Params
        self.beta = beta
        self.gamma = gamma
        self.dtype = dtype
        self.scale = scale
        self.k = k

    @property
    def x(self) -> np.ndarray:
        """Get the X-coordinates of vectors, considering interpolation.

        Returns
        -------
            np.ndarray: X-coordinates.

        """
        if self.V is None:
            return self._x
        else:
            j = np.indices(self.V.shape)[1,:,:,0]
            return j.reshape(-1)

    @property
    def y(self) -> np.ndarray:
        """Get the Y-coordinates of vectors, considering interpolation.

        Returns
        -------
            np.ndarray: Y-coordinates.

        """
        if self.V is None:
            return self._y
        else:
            i = np.indices(self.V.shape)[0,:,:,0]
            return i.reshape(-1)

    @property
    def u(self) -> np.ndarray:
        """Get the horizontal components of vectors, considering interpolation.

        Returns
        -------
            np.ndarray: Horizontal components.

        """
        if self.V is None:
            return self._u
        else:
            return self.V[:,:,1].reshape(-1)

    @property
    def v(self) -> np.ndarray:
        """Get the vertical components of vectors, considering interpolation.

        Returns
        -------
            np.ndarray: Vertical components.

        """
        if self.V is None:
            return self._v
        else:
            return self.V[:,:,0].reshape(-1)


    def plot(self, fig: Figure = None, ax: Axes = None, *, show: bool = True,
             ) -> Figure:
        """Plot the vectors.

        Args:
        ----
            fig (Figure, optional): Matplotlib Figure object to use for
            plotting. Defaults to None.
            ax (Axes, optional): Matplotlib Axes object to use for plotting.
            Defaults to None.
            show (bool, optional): Whether to display the plot. Defaults to
            True.

        Returns:
        -------
            Figure: Matplotlib Figure object.

        """
        if (fig is None) & (ax is None):
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.invert_yaxis()

        ax.quiver(self.x, self.y, self.u, self.v, angles='xy')

        if show:
            fig.show()

        return fig

    def check_positiveness(self) -> np.ndarray:
        """Check if all the matrices in the `H` attribute are positive definite.

        Returns:
        -------
            positiveness (ndarray):
                A boolean array of shape `(ny_, nx_)` where `ny_` and `nx_`
                are the number of rows and columns in the `H` attribute
                respectively.
                It indicates whether each matrix in `H` is positive definite
                or not.

        """
        nx_ = self.H.shape[1]
        ny_ = self.H.shape[0]

        positiveness = np.zeros((ny_, nx_),dtype=bool)

        for j in range(nx_):
            for i in range(ny_):
                x = self.H[i, j, :, :]
                positiveness[i, j] = np.all(np.linalg.eigvals(x) > 0)

        return positiveness

    def check_differenciable(self, dx: int, dy: int) -> np.ndarray:
        """Check if H is differenciable.

        Parameters
        ----------
        dx : int
            The increment in the x-direction.
        dy : int
            The increment in the y-direction.

        Returns
        -------
        ndarray
            A 4D array representing the partial derivatives of H with respect
            to x and y.

        """
        nx_ = self.H.shape[1]
        ny_ = self.H.shape[0]

        differenciable = np.zeros((ny_, nx_, 2, 2))

        for j in range(1,nx_-1):
            for i in range(1, ny_ - 1):

                dh_dy = (self.H[i + 1, j, :, :] - self.H[i - 1, j, :, :]
                         ) / (2 * dy)
                dh_dx = (self.H[i, j + 1, :, :] - self.H[i , j - 1, :, :]
                         ) / (2 * dx)

                dh00_dy = dh_dy[0, 0]
                dh01_dy = dh_dy[0, 1]
                dh11_dx = dh_dx[1, 1]
                dh10_dx = dh_dx[1, 0]


                differenciable[i, j, 0, 0] = dh00_dy
                differenciable[i, j, 0, 1] = dh01_dy
                differenciable[i, j, 1, 1] = dh11_dx
                differenciable[i, j, 1, 0] = dh10_dx

        return differenciable

def anisotropy_from_svg(path: str, beta: float, gamma: float,
                        norm_type: str = 'norm', k: int = 3) -> Anisotropy:
    """Create an Anisotropy object from an SVG file.

    Args:
    ----
        path (str): Path to the SVG file.
        beta (float): Beta parameter.
        gamma (float): Gamma parameter.
        norm_type (str, optional): Type of normalization. Defaults to 'norm'.
        k (int, optional): No of neighbours to consider when interpolating.
        Defaults to 3.

    Returns:
    -------
        Anisotropy: An Anisotropy object.

    """
    tree = Et.parse(path)
    root = tree.getroot()

    xmlns = get_inkscape_namespace(root)
    tfm = get_inkscape_transform(root, xmlns)

    view_box = root.attrib['viewBox']
    view_box = re.findall(r'(-?[\d]+[\.]?[\d]*)', view_box)
    _, _, width, height = (float(i) for i in view_box)

    P = []

    # Remove defitinions to skip style 'path' elements, like arrows heads.
    defs = next(root.iter(f'{xmlns}defs'))
    root.remove(defs)

    for child in root.iter(f'{xmlns}path'):

        values = child.attrib['d'][2:]
        style = child.attrib['d'][0]

        p0, p1 = values.split(' ')

        p0 = p0.split(',')
        p1 = p1.split(',')

        p0 = [float(p0[0]), float(p0[1])]
        p1 = [float(p1[0]) , float(p1[1])]

        # m style is incremental / M style is absolute
        if style == 'M':
            p1 = [p1[0] - p0[0], p1[1] - p0[1]]

        # Flip anchor point (p0) and incremental point (p1)
        p0[1] =  p0[1]
        p1[1] = p1[1]

        P.append([p0[0], p0[1], p1[0], p1[1]])

    P = np.asarray(P)

    x = P[:,:2] +  tfm
    u = P[:,2:]

    norm = (u**2).sum(axis=1)**0.5
    if norm_type == 'min':
        min_norm = (norm).min()
        u = u/(min_norm)
    elif norm_type == 'max':
        max_norm = (norm).max()
        u = u/(max_norm)
    elif norm_type == 'norm':
        u = u/norm[:, np.newaxis]
    else:
        msg = f'Unknown norm_type = "{norm_type}"'
        raise ValueError(msg)

    x_u = np.column_stack((x,u))

    return Anisotropy(x_u, beta, gamma, width,
                      height, k)


def anisotropy_stationary(theta: float, gamma: float, beta: float,
                          ) -> Anisotropy:
    """Create an Anisotropy stationary anisotrpy from an angle.

    Parameters
    ----------
    theta : float
        The angle in degrees at which the anisotropy pattern is oriented.
    gamma : float
        The isotropic baseline effect coefficient.
    beta : float
        The anisotropic local effect coefficient.

    Returns
    -------
    Anisotropy
        An Anisotropy object representing the stationary anisotropy pattern.

    """
    theta = np.radians(theta)

    v =  np.asarray([- np.sin(theta),  np.cos(theta)])
    x_u  = np.asarray([[0,0,v[0],v[1]]])

    return Anisotropy(x_u, beta, gamma, scale=False)
