
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import xml.etree.ElementTree as Et
from matplotlib.figure import Figure

__all__ = ['Anisotropy', 'anisotropy_from_svg']

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
                 width: float, height: float, dtype: type = np.float64,
                 ) -> None:
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
            dtype (type): Data type for the fields tensors.
            Defaults to np.float32.

        """
        # No Interpolation
        self._x = x_u[:,0].astype(dtype)
        self._y = x_u[:,1].astype(dtype)
        self._u = x_u[:,2].astype(dtype)
        self._v = x_u[:,3].astype(dtype)
        self.width = width
        self.height = height

        # Interpolated
        self.H = None
        self.V = None

        # Params
        self.beta = beta
        self.gamma = gamma
        self.dtype = dtype

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

def anisotropy_from_svg(path: str, beta: float, gamma: float,
                        norm_type: str = 'norm') -> Anisotropy:
    """Create an Anisotropy object from an SVG file.

    Args:
    ----
        path (str): Path to the SVG file.
        beta (float): Beta parameter.
        gamma (float): Gamma parameter.
        norm_type (str, optional): Type of normalization. Defaults to 'norm'.

    Returns:
    -------
        Anisotropy: An Anisotropy object.

    """
    tree = Et.parse(path)
    root = tree.getroot()

    xmlns = re.search(r'({.*})',root.tag)
    xmlns = xmlns.group(1) if xmlns is not None else ''

    width = float(root.attrib['width'][:-2])
    height = float(root.attrib['height'][:-2])

    P = []

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

        # When in quadrant III and IV, switch to quadrant I or II.
        if p1[0] < 0:
            p1[0] *= -1
            p1[1] *= -1

        P.append([p0[0], p0[1], p1[0], p1[1]])

    P = np.asarray(P)

    x = P[:,:2]
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
                      height)
