import sys, os
sys.path.append('../')

import numpy as np
from typing import List

from used.CreatPWithParameter import CreatePWithParameter
from used.pre_process import kl_diver

if __name__ == '__main__':

    """
    pz pd_z pw_z を取得
    """
    plsa_z_size = 15
    print("plsa_z_size" + str(plsa_z_size))

    save_path_pz: str = os.getcwd() + "/data/pz/estimated_pz_z" + str(plsa_z_size) + ".txt"
    try:
        with open(save_path_pz) as f:
           pz = list(map(float, f.readline().split()))
           pz: np.ndarray = np.array(pz)

    except(FileNotFoundError):
        print("/data/pz/estimated_pz_z" + str(plsa_z_size) + ".txt not found")
        exit()

    save_path_pw: str = os.getcwd() + "/data/pw/estimated_pw_z" + str(plsa_z_size) + ".txt"
    try:
        with open(save_path_pw) as f:
            pw_z: List[int] = []
            while True:
                line = (f.readline()).strip()
                if line == 'end':
                    break
                pw_z.append(list(map(float, line.split())))
            pw_z: np.ndarray= np.array(pw_z)
    except(FileNotFoundError):
        print("/data/pw/estimated_pw_z" + str(plsa_z_size) + ".txt not found")
        exit()


    save_path_pd: str = os.getcwd() + "/data/pd/estimated_pd_z" + str(plsa_z_size) + ".txt"
    try:
        with open(save_path_pd) as f:
            pd_z: List[int] = []
            while True:
                line = (f.readline()).strip()
                if line == 'end':
                    break
                pd_z.append(list(map(float, line.split())))
            pd_z: np.ndarray = np.array(pd_z)
    except(FileNotFoundError):
        print("/data/pd/estimated_pd_z" + str(plsa_z_size) + ".txt not found")
        exit()

    estimated_pwd: np.ndarray = np.dot(np.dot(pw_z.T, np.diag(pz)), pd_z)
    #print(estimated_pwd.shape)
    """
     plsaからのp(w, d)と生成したp(w, d)のKLdiverをとる
    """

    # alpha beta gannmaを取得
    path = os.getcwd() + "/parameter_pd.txt"
    p = CreatePWithParameter(path)

    # p(w, d)を作成する回数
    K: int = 100

    # klの平均
    mean_of_kl: int = 0

    for i in range(K):
        if (i+1) % 10 == 0:
            print(str(i+1) + "回目")
        pwd = p.make_p(pd_z.shape[1])
        mean_of_kl += kl_diver(pwd, estimated_pwd)

    mean_of_kl /= K
    print(mean_of_kl)



