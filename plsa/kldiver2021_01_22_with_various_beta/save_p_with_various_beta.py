import numpy as np
import sys, os
from typing import List
sys.path.append('../')

from used.CreatP import CreatP

if __name__ == "__main__":

    path = os.getcwd() + "/parameter_with_various_beta.txt"
    try:
        with open(path) as f:
            while True:
                line = (f.readline()).strip()
                if line == "alpha":
                    alpha = list(map(float, (f.readline()).split()))
                    alpha = np.array(alpha)

                if line == "beta":
                    beta: List[List[float]] = []
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

    # ディリクレ分布はわかったが乱数の生成法はわからない
    Pd = np.random.dirichlet(ganma)  # diricle(gamnma) d_size
    w_size = beta.shape[1]
    z_size = len(alpha)

    # Pz_d P(z|d)
    tmp_ls = []
    for _ in range(len(ganma)):
        tmp_dir = np.random.dirichlet(alpha)  # diricle(alpha) z
        tmp_ls.append(tmp_dir)

    Pz_d = np.array(tmp_ls)

    tmp_ls = []
    # phi[i][j] i番目のトッピックにおいてj番目の語彙が出る確率
    for i in range(z_size):
        tmp_dir = np.random.dirichlet(beta[i])  # diricle(beta) w_size
        tmp_ls.append(tmp_dir)
    phi = np.array(tmp_ls)

    print(Pd.shape)
    print()
    print(Pz_d.shape)
    print()
    print(phi.shape)

    path = os.getcwd() + "/parameter_pd_with_various_beta.txt"
    try:
        with open(path, mode="x") as f:
            f.write('pd \n')
            pd: List[int] = list(map(str, list(Pd)))
            f.write(' '.join(pd))
            f.write('\n')
            f.write('\n')

            f.write('pz_d \n')
            for pz_d in Pz_d:
                pz: List[int] = list(map(str, list(pz_d)))
                f.write(' '.join(pz))
                f.write('\n')

            f.write('end\n')
            f.write('\n')

            f.write('phi \n')
            for ph in phi:
                p: List[int] = list(map(str, list(ph)))
                f.write(' '.join(p))
                f.write('\n')

            f.write('end\n')
    except(FileExistsError):
        print('file exit')
        exit()

