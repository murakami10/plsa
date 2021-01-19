import os

from typing import List
import numpy as np

path = os.getcwd() + "/parameter_pd_with_various_beta.txt"
with open(path) as f:
    while True:
        line = (f.readline()).strip()

        if line == 'pd':
            Pd = list(map(float, f.readline().split()))
            Pd = np.array(Pd)

        if line == 'pz_d':
            Pz_d: List[int] = []
            while True:
                line = (f.readline()).strip()
                if line == 'end':
                    break
                Pz_d.append(list(map(float, line.split())))
            Pz_d = np.array(Pz_d)

        if line == 'phi':
            phi: List[int] = []
            while True:
                line = (f.readline()).strip()
                if line == 'end':
                    break
                phi.append(list(map(float, line.split())))

            phi: np.ndarray = np.array(phi)
            break

pw_zpz_d: np.ndarray = (np.dot(Pz_d, phi)).T
p_wd: np.ndarray = np.tile(Pd, (phi.shape[1], 1)) * pw_zpz_d

path = os.getcwd() + "/true_pwd.txt"
with open(path, mode="w") as f:
    for p in p_wd:
        p_str = list(map(str, p))
        f.write(" ".join(p_str))
        f.write("\n")
    f.write("end")
