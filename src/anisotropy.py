import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import xml.etree.ElementTree as ET
from matplotlib.figure import Figure


class Anisotropy:

    def __init__(self, x: np.ndarray, y: np.ndarray , u: np.ndarray, 
                 v: np.ndarray, beta: float, gamma: float, width: float, 
                 height: float, dtype: type = np.float32):

        # No Interpolation
        self._x = x.astype(dtype)
        self._y = y.astype(dtype)
        self._u = u.astype(dtype)
        self._v = v.astype(dtype)
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
    def x(self):
        if self.V is None:
            return self._x
        else:
            j = np.indices(self.V.shape)[1,:,:,0]
            return j.reshape(-1)

    @property
    def y(self):
        if self.V is None:
            return self._y
        else:
            i = np.indices(self.V.shape)[0,:,:,0]
            return i.reshape(-1)

    @property
    def u(self):
        if self.V is None:
            return self._u
        else:
            return self.V[:,:,1].reshape(-1)
        
    @property
    def v(self):
        if self.V is None:
            return self._v
        else:
            return self.V[:,:,0].reshape(-1)


    def plot(self, show: bool = True, fig: Figure = None, ax: Axes = None):

        if (fig is None) & (ax is None):
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.invert_yaxis()

        ax.quiver(self.x, self.y, self.u, self.v, angles='xy')

        if show:
            fig.show()

        return fig


def from_svg(path, beta: float, gamma: float, norm_type: str = 'norm'):

    tree = ET.parse(path)
    root = tree.getroot()

    width = float(root.attrib['width'][:-2])
    height = float(root.attrib['height'][:-2])

    P = []
    for child in root:
        if child.tag.endswith('path'):
            values = child.attrib['d'][2:]
            style = child.attrib['d'][0]

            p0, p1 = values.split(' ')

            p0 = p0.split(',')
            p1 = p1.split(',')

            p0 = [float(p0[0]), float(p0[1])]
            p1 = [float(p1[0]) , float(p1[1])]

            # m style is incrementlal / M style is absolute
            if style == 'M':
                p1 = [p1[0] - p0[0], p1[1] - p0[1]]

            # Flip anchor point (p0) and increment point (p1)
            p0[1] =  p0[1]
            p1[1] = p1[1]

            # When in quanrdrant III and IV, sqitch to quandran I or II.
            if p1[0] < 0:
                p1[0] *= -1
                p1[1] *= -1

            P.append([p0[0], p0[1], p1[0], p1[1]])

    P = np.asarray(P)
    x = P[:,:2]
    y = P[:,2:]

    norm = (y**2).sum(axis=1)**0.5

    if norm_type == 'min':
        min_norm = (norm).min()
        y = y/(min_norm)
    elif norm_type == 'max':
        max_norm = (norm).max()
        y = y/(max_norm)
    elif norm_type == 'norm':
        y = y/norm[:, np.newaxis]
    else:
        raise ValueError(f'Unknown norm_type = "{norm_type}"')

    return Anisotropy(x[:, 0], x[:, 1], y[:, 0], y[:, 1], beta, gamma, width, height)

