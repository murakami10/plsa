import numpy as np
import os

path = os.getcwd() + "/parameter_with_various_beta.txt"

# 理論値のpz pd_z pw_zを計算する

try:
    with open(path) as f:
        while True:
            line = (f.readline()).strip()
            if line == "alpha":
                alpha = list(map(float, (f.readline()).split()))
                alpha = np.array(alpha)

            if line == "beta":
                beta = []
                while True:
                    line = (f.readline()).strip()
                    if line == "end":
                        break
                    beta.append(list(map(float, line.split())))
                beta: np.ndarray = np.array(beta)

            if line == "ganma":
                ganma = list(map(float, (f.readline()).split()))
                ganma = np.array(ganma)
                break
except FileNotFoundError:
    print("file not found")
    exit()


try:

    pz = alpha/sum(alpha)
    pz = list(map(str, pz))
    with open(path, mode='a') as f:
        f.write("pz \n")
        f.write(" ".join(pz))
        f.write("\n\n")

    with open(path, mode="a") as f:
        f.write("pw_z \n")
        for bet in beta:
            pw_z = bet/sum(bet)
            pw_z = list(map(str, pw_z))
            f.write(" ".join(pw_z))
            f.write("\n")
        f.write("\n")

    pd_z = ganma/sum(ganma)
    pd_z = list(map(str, pd_z))
    with open(path, mode="a") as f:
        f.write("pd_z \n")
        f.write(" ".join(pd_z))
        f.write("\n\n")
except FileExistsError:
    print("file already exists")
    exit()
