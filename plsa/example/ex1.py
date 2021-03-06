import numpy as np
import sys

sys.path.append('../')

from used.Plsa import Plsa
from used.pre_process import kl_diver, kl_diver_with_label
'''

N = np.array([
    [20, 23, 1, 4],
    [25, 19, 3, 0],
    [2, 1, 31, 28],
    [0, 1, 22, 17],
    [1, 0, 18, 24]
])

plsa = Plsa(N, 3)
plsa.train(k=1)
z = plsa.Pz
x = plsa.Px_z
W = np.dot(x.T, np.diag(z))
H = plsa.Py_z
pwd = np.dot(W, H)
print(plsa.Pz)
print(pwd)

plsa.sort_pz_px_py()
z = plsa.Pz
x = plsa.Px_z
W = np.dot(x.T, np.diag(z))
H = plsa.Py_z
pwd = np.dot(W, H)
print(plsa.Pz)
print(pwd)
'''

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

N = np.array([
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0]
])
N /= np.sum(N)

N1 = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 0.0]
])
label = [0, 0, 0, 0]

N1 /= np.sum(N1)

print(kl_diver_with_label(N, N1, label))
'''
N = np.array([
    [1.0, 0.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0]
])
print(np.any(N == 0))
'''
