import numpy as np
import os

path = os.getcwd() + "/parameter_with_various_alpha_beta.txt"

# alpha beta gannmaを作成 betaはバラバラ

try:

    # len(alpha) がトピックの数　
    alpha_len = 10
    alpha_num = 50
    # alpha =
    with open(path, mode='x') as f:
        f.write("alpha \n")
        for i in range(alpha_num):
            alpha = np.random.rand(alpha_len)
            alpha *= 5
            for j in range((i//(alpha_num//alpha_len)), (i//(alpha_num//alpha_len))+1):
                alpha[j] += 20
            alpha = list(map(str, alpha))
            f.write(" ".join(alpha))
            f.write("\n")
        f.write("end\n\n")


    # len(beta) が語彙の数　
    beta_len = 210
    beta_num = 10
    # beta =
    with open(path, mode="a") as f:
        f.write("beta \n")
        for i in range(beta_num):
            beta = np.random.rand(beta_len)
            beta *= 5
            for j in range((beta_len//beta_num) * i, (beta_len//beta_num) * (i+1)):
                beta[j] += 20
            beta = list(map(str, beta))
            f.write(" ".join(beta))
            f.write("\n")
        f.write("end\n\n")

    # len(ganma) が文書の種類
    ganma_len = 50
    divide = 10
    # ganma
    ganma = np.random.rand(ganma_len)
    ganma *= 3
    for i in range(ganma_len):
        ganma[i] += 5*(i//(ganma_len//divide))
    ganma = list(map(str, ganma))
    with open(path, mode="a") as f:
        f.write("ganma \n")
        f.write(" ".join(ganma))
        f.write("\n\n")
except FileExistsError:
    print("file already exists")
    exit()
