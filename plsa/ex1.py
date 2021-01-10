import numpy as np
from used.Plsa import Plsa


N = np.array([
    [20, 23, 1, 4],
    [25, 19, 3, 0],
    [2, 1, 31, 28],
    [0, 1, 22, 17],
    [1, 0, 18, 24]
])

plsa = Plsa(N, 2)
plsa.train(k=1)

'''
print('P(z)')
print(plsa.Pz)
print('P(x|z)')
print(plsa.Px_z)
print('P(y|z)')
print(plsa.Py_z)
print('P(z|x)')
Pz_x = plsa.Px_z.T * plsa.Pz[None, :]
print(Pz_x / np.sum(Pz_x, axis=1)[:, None])
print('P(z|y)')
Pz_y = plsa.Py_z.T * plsa.Pz[None, :]
print(Pz_y / np.sum(Pz_y, axis=1)[:, None])
'''

