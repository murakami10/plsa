import numpy as np
from typing import List
import sys
sys.path.append('../')
from used.plsa import PLSA
from used.pre_process import zentai, kl_diver

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
    alpha = [1, 1, 1]  # len(alpha)がトピックの数
    beta = [2, 2, 2, 2]  # len(beta)が語彙の数
    ganma = [2, 2, 2, 2]  # len(ganma)が文書の集合の個数
    d = 4 # p(d, w) のdの数
    #plsaのzはどうするのか
    plsa_z = 3

    c = CreatP(alpha, beta, ganma)
    p = c.make_p(d)

    #print(zentai(p))
    #print()
    #print(p)
    plsa = PLSA(p, plsa_z)
    plsa.train(10000)

    after_p = np.dot(np.dot(plsa.Px_z.T, np.diag(plsa.Pz)), plsa.Py_z)
    #print(after_p)
    #print()

    print(kl_diver(p, after_p))

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

