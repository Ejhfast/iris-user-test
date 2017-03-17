import fileinput
from collections import defaultdict
import numpy as np
import pickle

cats = defaultdict(list)

for i,line in enumerate(fileinput.input()):
    if i == 0:
        header = line.strip().split(",")
    else:
        cols = line.strip().split(",")
        for j,c in enumerate(cols):
            cats[header[j]].append(c)

for k,vs in cats.items():
    if k != "Name":
        cats[k] = np.array([float(x) for x in vs])
    else:
        k2i = {k:i for i,k in enumerate(sorted(set(vs)))}
        print(k2i)
        cats[k] = np.array([k2i[x] for x in vs])

with open('iris-dict.pkl', 'wb') as f:
    pickle.dump(cats, f)
