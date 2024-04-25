from scipy import sparse

columns = 500
rows = 500

def precreated_list():

    row, col, val = [], [], []

    for i in range(rows):
        for j in range(columns):
            row.append(i)
            col.append(j)
            val.append(1)

    breakpoint()
    sparse.csc_matrix((val, (row, col)))


precreated_list()