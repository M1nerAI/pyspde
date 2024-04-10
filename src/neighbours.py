

class Neighbours:
    _neighs = {'id': {'i': 0, 'j': 0},
               'id_left': {'i': 0, 'j': -1},
               'id_top_left': {'i': -1, 'j': -1},
               'id_top': {'i': -1, 'j': 0},
               'id_top_right': {'i': -1, 'j': 1},
               'id_right': {'i': 0, 'j': 1},
               'id_bot_right': {'i': 1, 'j': 1},
               'id_bot': {'i': 1, 'j': 0},
               'id_bot_left': {'i': 1, 'j': -1}}
    
    def __init__(self, i, j, func):
        self.i = i
        self.j = j

        for pos, delta in self._neighs.items():
            setattr(self, pos, func(i + delta['i'], j + delta['j']))

        self.allowable_neighs = [k for k, v in self.__dict__.items()
                                 if k.startswith('id_') and v is not None]


    def __repr__(self) -> str:
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