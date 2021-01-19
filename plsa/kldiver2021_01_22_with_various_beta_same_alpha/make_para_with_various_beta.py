import numpy as np
import os

# mini path
# path = os.getcwd() + "/parameter_with_various_beta.txt"

# large path
path = os.getcwd() + "/large_parameter_with_various_beta.txt"

# alpha beta gannmaを作成 betaはバラバラ

try:

    # len(alpha) がトピックの数　
    alpha_len = 10
    # alpha =
    alpha = np.ones(alpha_len)
    alpha *= 20
    alpha = list(map(str, alpha))
    with open(path, mode='x') as f:
        f.write("alpha \n")
        f.write(" ".join(alpha))
        f.write("\n\n")


    # len(beta) が語彙の数　
    beta_len = 1000 # large_beta_len 1000 mini_beta_len 210
    beta_num = 10
    # beta =
    with open(path, mode="a") as f:
        f.write("beta \n")
        for i in range(beta_num):
            beta = np.random.rand(beta_len)
            beta *= 5
            for j in range((beta_len//beta_num) * i, (beta_len//beta_num) * (i+1)):
                beta[j] += 50
            beta = list(map(str, beta))
            f.write(" ".join(beta))
            f.write("\n")
        f.write("end\n")
        f.write("\n")

    # len(ganma) が文書の種類
    ganma_len = 200 # large_ganma_len 200 mini_ganma_len 50
    # ganma
    ganma = np.random.rand(ganma_len)
    ganma *= 10
    ganma = list(map(str, ganma))
    with open(path, mode="a") as f:
        f.write("ganma \n")
        f.write(" ".join(ganma))
        f.write("\n\n")
except FileExistsError:
    print("file already exists")
    exit()
