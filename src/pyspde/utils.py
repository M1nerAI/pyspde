from __future__ import annotations

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

    return nx*i + j


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
