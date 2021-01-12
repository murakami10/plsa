import numpy as np
import os
from used import CreatP
import sys
sys.path.append('../')
from used.Plsa import Plsa
from used.pre_process import zentai, process_result

path = os.getcwd() + "/parameter.txt"

try:
    with open(path) as f:
        while True:
            line = (f.readline()).strip()
            if line == "alpha":
                alpha = list(map(float, (f.readline()).split()))
                alpha = np.array(alpha)

            if line == "beta":
                beta = list(map(float, (f.readline()).split()))
                beta = np.array(beta)

            if line == "ganma":
                ganma = list(map(float, (f.readline()).split()))
                ganma = np.array(ganma)
                break
except FileNotFoundError:
    print("file not found")
    exit()

# len(alpha) がトピックの数　
# print(alpha.shape)
# len(beta) が語彙の数　
# print(beta.shape)
# len(ganma) が文書の種類
# print(ganma.shape)

for i in range(90):
    print(i+1, "回目")
    document_size = len(ganma)

    c = CreatP.CreatP(alpha, beta, ganma)
    p = c.make_p(document_size)
    p = zentai(p)

    PLSA_SIZE = 15
    TRAIN_NUM = 100000

    plsa = Plsa(p, PLSA_SIZE)
    plsa.train(TRAIN_NUM)

    path = os.getcwd() + "/data/pw_z/" + "d" + str(len(ganma)) + "w" + str(len(beta)) + "z" + str(len(alpha)) + "plsaz" + str(PLSA_SIZE) + ".txt"
    px_z = process_result(plsa.Px_z)
    print(px_z.shape)
    px_z = list(map(str, px_z))
    with open(path, mode="a") as f:
        f.write(" ".join(px_z))
        f.write("\n\n")

    path = os.getcwd() + "/data/pd_z/" + "d" + str(len(ganma)) + "w" + str(len(beta)) + "z" + str(len(alpha)) + "plsaz" + str(PLSA_SIZE) + ".txt"
    py_z = process_result(plsa.Py_z)
    print(py_z.shape)
    py_z = list(map(str, py_z))
    with open(path, mode="a") as f:
        f.write(" ".join(py_z))
        f.write("\n\n")

    path = os.getcwd() + "/data/pz/" + "d" + str(len(ganma)) + "w" + str(len(beta)) + "z" + str(len(alpha)) + "plsaz" + str(PLSA_SIZE) +".txt"
    pz = plsa.Pz
    print(pz.shape)
    pz = list(map(str, pz))
    with open(path, mode="a") as f:
        f.write(" ".join(pz))
        f.write("\n\n")
