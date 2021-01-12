import numpy as np
from typing import List
import sys, os
sys.path.append('../')
from used.Plsa import Plsa
from used.pre_process import zentai, kl_diver, process_result

class CreatP:
    def __init__(self, alpha: List[int], beta: List[int], ganma: List[int]):

        #ディリクレ分布はわかったが乱数の生成法はわからない
        self.Pd = np.random.dirichlet(ganma)# diricle(gamnma) d_size
        self.w_size = len(beta)
        self.z_size = len(alpha)

        #Pz_d P(z|d)
        tmp_ls = []
        for _ in range(len(ganma)):
            tmp_dir = np.random.dirichlet(alpha)# diricle(alpha) z
            tmp_ls.append(tmp_dir)

        self.Pz_d = np.array(tmp_ls)

        tmp_ls = []
        # phi[i][j] i番目のトッピックにおいてj番目の語彙が出る確率
        for _ in range(self.z_size):
            tmp_dir = np.random.dirichlet(beta)# diricle(beta) w_size
            tmp_ls.append(tmp_dir)
        self.phi = np.array(tmp_ls)

    def make_p(self, d_size: int) -> np.ndarray:
        # d_siezはここで選ぶ？#####################################################
        p = [[0]*d_size for _ in range(self.w_size)]
        for d in range(d_size):

            di = np.random.choice(len(self.Pd), p=self.Pd) # Pdから文書を選択
            #print(di)
            Ndi = np.random.poisson(self.w_size * 2)
            #print(Ndi)
            for _ in range(Ndi):
                z = np.random.choice(self.z_size, p=self.Pz_d[di])
                w = np.random.choice(self.w_size, p=self.phi[z])
                p[w][d] += 1
        return np.array(p)

if __name__ == '__main__':
    #alpha beta ganmaは一桁
    alpha = [1, 20, 1, 20]  # len(alpha)がトピックの数
    beta = [2, 1, 2, 1, 2, 1, 1, 2]  # len(beta)が語彙の数
    ganma = [1, 20, 1, 20, 1]  # len(ganma)が文書の集合の個数
    # print(len(beta))
    # print(len(ganma))

    d = 5 # p(d, w) のdの数
    # plsaのzはどうするのか
    plsa_z = 4

    c = CreatP(alpha, beta, ganma)
    p = c.make_p(d)

    p = zentai(p)
    # print(p)
    plsa = Plsa(p, plsa_z)
    plsa.train(1000)

    print(plsa.Py_z)
    py_z = process_result(plsa.Py_z)
    print(py_z)

    #after_p = np.dot(np.dot(plsa.Px_z.T, np.diag(plsa.Pz)), plsa.Py_z)
    #print(after_p)
    #print()

    #print(kl_diver(p, after_p))

    '''
    print('Pd')
    print(c.Pd)
    print()
    print('Pz_d')
    print(c.Pz_d)
    print()
    print('phi')
    print(c.phi)
    '''
