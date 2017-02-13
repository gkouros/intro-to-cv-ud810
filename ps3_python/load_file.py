import numpy as np

def load_file(name):
    lst = []
    f = open(name, 'r')
    for line in f:
        lst.append(line.strip().split())
    f.close()
    return np.array(lst, dtype=np.float32)

