import numpy as np
import os

path = os.getcwd() + "/parameter.txt"


try:

    # len(alpha) がトピックの数　
    # alpha =
    alpha = np.random.rand(10)
    alpha *= 10
    alpha = list(map(str, alpha))
    with open(path, mode='x') as f:
        f.write("alpha \n")
        f.write(" ".join(alpha))
        f.write("\n\n")


    # len(beta) が語彙の数　
    # beta =
    beta = np.random.rand(210)
    beta *= 10
    beta = list(map(str, beta))
    with open(path, mode="a") as f:
        f.write("beta \n")
        f.write(" ".join(beta))
        f.write("\n\n")

    # len(ganma) が文書の種類
    # ganma
    ganma = np.random.rand(50)
    ganma *= 10
    ganma = list(map(str, ganma))
    with open(path, mode="a") as f:
        f.write("ganma \n")
        f.write(" ".join(ganma))
        f.write("\n\n")
except FileExistsError:
    print("file already exists")
    exit()
