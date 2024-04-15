from .utils import imdict
from dataclasses import dataclass
from typing import Callable

class Neighbours:
    """Neighbours Class.

    Class containing the flatten index of the 8 neighbouring elements in
    the 2d grid.

    Attributes
    ----------
    i : int
        The row index of the cell.
    j : int
        The column index of the cell.
    func : function
        A function used to calculate values for neighboring cells.
    allowable_neighs : list
        List of allowable neighboring cell indices.

    Methods
    -------
    __repr__()
        Returns a string representation of the neighboring cell indices and XY
        coordinates.

    """

    _neighs = imdict(id=imdict(i=0, j=0),
                     id_left=imdict(i=0, j=-1),
                     id_top_left=imdict(i=-1, j=-1),
                     id_top=imdict(i=-1, j=0),
                     id_top_right=imdict(i=-1, j=1),
                     id_right=imdict(i=0, j=1),
                     id_bot_right=imdict(i=1, j=1),
                     id_bot=imdict(i=1, j=0),
                     id_bot_left=imdict(i=1, j=-1))

    def __init__(self, i: int, j: int, func: Callable) -> None:
        """Initialize the Neighbours class.

        Parameters
        ----------
        i : int
            Row index.
        j : int
            Column index.
        func : Callable
            Index mapping function.

        """
        self.i = i
        self.j = j

        for pos, delta in self._neighs.items():
            setattr(self, pos, func(i + delta['i'], j + delta['j']))

        self.allowable_neighs = [k for k, v in self.__dict__.items()
                                 if k.startswith('id_') and v is not None]

    def has(self, pos: str) -> bool:
        """Check if neighbour exists.

        Parameters
        ----------
        pos : str
            Neighbour to check.

        Returns
        -------
        bool
            A boolean indicating whether the neighbour exists.

        """
        return getattr(self, pos, None) is not None



    def __repr__(self) -> str:
        """Return a schematic view of neighbours.

        For both 1D Flatten index and 2D XY Coordinates.
        """
        neighs = {}
        for neigh in self._neighs:
            value = getattr(self, neigh)
            if value is None:
                neighs[neigh] =' '*6
            else:
                neighs[neigh] = f'{value:06d}'

        ids = \
        "1D Flatten index:\n"+\
        "_"*27 + "\n"+\
        f"|{neighs['id_top_left']} | {neighs['id_top']} | {neighs['id_top_right']}|\n"+\
        f"|{neighs['id_left']} | {neighs['id']} | {neighs['id_right']}|\n"+\
        f"|{neighs['id_bot_left']} | {neighs['id_bot']} | {neighs['id_bot_right']}|\n"+\
        "‾"*27 + "\n\n"


        neighs = {}
        for neigh, delta in self._neighs.items():
            value = getattr(self, neigh)
            if value is None:
                neighs[neigh] =' '*11
            else:
                neighs[neigh] = f"({self.j + delta['j']:>4},"\
                                f"{self.i + delta['i']:>4})"

        xy = \
        "2D XY Coordinates:\n"+\
        "_"*41 + "\n"+\
        f"|{neighs['id_top_left']} | {neighs['id_top']} | {neighs['id_top_right']}|\n"+\
        f"|{neighs['id_left']} | {neighs['id']} | {neighs['id_right']}|\n"+\
        f"|{neighs['id_bot_left']} | {neighs['id_bot']} | {neighs['id_bot_right']}|\n"+\
        "‾"*41 + "\n"

        return ids + xy
