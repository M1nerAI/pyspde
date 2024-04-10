
def map_2d_1d_nx_ny(i, j, nx, ny):

    if (i < 0) or (j < 0) or (j >= nx) or (i >= ny):
        return None

    return nx*i + j


def map_1d_2d_nx(idx, nx):
    j = int(idx % nx)
    i = int((idx - j)/nx)
    return i, j
