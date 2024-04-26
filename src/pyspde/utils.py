from __future__ import annotations
import re

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xml.etree.ElementTree as Et

__all__ = ['sample_grid']

class imdict(dict):
    def __hash__(self) -> None:
        """Return the unique id of the object as its hash value."""
        return id(self)

    def _immutable(self, *args, **kws) -> None:
        """Raise TypeError for methods that attempt mutability."""
        msg = 'Object is immutable'
        raise TypeError(msg)

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear       = _immutable
    update      = _immutable
    setdefault  = _immutable
    pop         = _immutable
    popitem     = _immutable


def map_2d_1d_nx_ny(i: int, j: int, nx: int, ny: int) -> int | None:
    """Map a 2D index to a 1D index given the dimensions of the 2D array.

    Parameters
    ----------
    i : int
        The row index.
    j : int
        The column index.
    nx : int
        The total width of the 2D array.
    ny : int
        The total height of the 2D array.

    Returns
    -------
    int or None
        The mapped 1D index if the input indices are within bounds, otherwise
        None.

    """
    if (i < 0) or (j < 0) or (j >= nx) or (i >= ny):
        return None

    return int(nx*i + j)


def map_1d_2d_nx(idx: int, nx: int) -> tuple[int, int]:
    """Map a 1D index to a 2D index given the total width of the 2D array.

    Parameters
    ----------
    idx : int
        The 1D index to be mapped.
    nx : int
        The total width of the 2D array.

    Returns
    -------
    Tuple[int, int]
        A tuple containing the mapped 2D indices (i, j).

    """
    j = int(idx % nx)
    i = int((idx - j) / nx)

    return i, j


def get_inkscape_transform(root: Et.Element, xmlns: str,
                           ) -> tuple[float, float]:
    """Get the inkscape transform from the root element.

    Parameters
    ----------
    root : Et.Element
        The root element.
    xmlns : str
        The namespace string.

    Returns
    -------
    Tuple[float, float]
        A tuple containing the x and y shift values.

    """
    try:
        translate = next(root.iter(f'{xmlns}g')).attrib['transform']

        translate = re.search(r'translate\((-?\d+\.\d+),(-?\d+\.\d+)\)',translate)
        if translate is None:
            translate = (0, 0)
        else:
            shift_x = float(translate.group(1))
            shift_y = float(translate.group(2))

            translate = (shift_x, shift_y)

    except Exception as e:
        print('Unknown error while getting inkscape transform: ', e)
        translate = (0, 0)

    return translate


def get_inkscape_namespace(root: Et.Element) -> str:
    """Extract the XML namespace from the root element tag.

    Parameters
    ----------
    root : ElementTree.Element
        The root element of the XML document.

    Returns
    -------
    str
        The XML namespace extracted from the root element tag. If no namespace 
        is found, an empty string is returned.

    """
    xmlns = re.search(r'({.*})',root.tag)
    xmlns = xmlns.group(1) if xmlns is not None else ''

    return xmlns


def sample_grid(Z: np.ndarray, dx: float, dy: float,
             ) -> np.ndarray:
    """Regular sampling from a 2D grid.

    Parameters
    ----------
    Z : np.ndarray
        The input array from which to sample points.
    dx : float
        The spacing between sample points in the x-direction.
    dy : float
        The spacing between sample points in the y-direction.

    Returns
    -------
    np.ndarray
        An array of sample points with shape (n, 3), where n is the number of
        sample points.

    """
    ny, nx, * _ = Z.shape

    sx = np.arange(0, nx, dx, dtype=int)
    sy = np.arange(0, ny, dy, dtype=int)

    xv, yv = np.meshgrid(sx, sy, indexing='xy')

    xv = xv.reshape(-1)
    yv = yv.reshape(-1)
    samps = Z[yv, xv]

    X = np.column_stack((xv, yv, samps))

    return X


def append_element(vals: list, rows: list, cols: list, val: float, row: int,
                   col: int) -> None:
    """Append an element for cosntructiong the sparse matrix.

    Parameters
    ----------
    vals : list
        The list to append the value to.
    rows : list
        The list to append the row index to.
    cols : list
        The list to append the column index to.
    val : float
        The value to append to the `vals` list.
    row : int
        The row index to append to the `rows` list.
    col : int
        The column index to append to the `cols` list.

    Returns
    -------
    None

    """
    vals.append(val)
    rows.append(row)
    cols.append(col)
