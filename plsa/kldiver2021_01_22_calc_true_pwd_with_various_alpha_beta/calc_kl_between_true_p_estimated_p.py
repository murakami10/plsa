import sys, os
sys.path.append('../')

import numpy as np
from typing import List

from used.CreatPWithParameter import CreatePWithParameter
from used.pre_process import zentai, kl_diver
from used.Plsa import Plsa

if __name__ == "__main__":

    plsazzz = [5, 10]
    for zzz in plsazzz:
        # plsaにおけるトピックの数
        plsa_z_size = zzz
        print(plsa_z_size)
        # plsaにおけるemステップの回数
        plsa_train_num = 100000
        # 作成するplsaの数
        estimate_num = 1

        # alpha beta gannmaを取得
        path = os.getcwd() + "/parameter_pd_with_various_alpha_beta.txt"
        p = CreatePWithParameter(path)

        # 真のp(w, d)を取得
        path = os.getcwd() + "/true_pwd.txt"
        with open(path) as f:
            true_pwd = []
            while True:
                line = (f.readline()).strip()
                if line == 'end':
                    break
                true_pwd.append(list(map(float, line.split())))
            true_pwd: np.ndarray = np.array(true_pwd)

        # tmp_klを初期化
        tmp_kl: float = 0
        tmp_kl_path: str = os.getcwd() + "/data/tmp_kl/z_" + str(plsa_z_size) + ".txt"
        estimated_pwd_path: str = os.getcwd() + "/data/estimated_pwd/z_" + str(plsa_z_size) + ".txt"
        for i in range(estimate_num):
            print(str(i + 1) + "回目")
            # トピック数をplsa_z_sizeとしてplsaを実行
            p_wd = p.make_p(p.Pd.shape[0])
            # p(w, d)が全体で1になるように
            p_wd = zentai(p_wd)
            plsa = Plsa(p_wd, plsa_z_size)
            # plsa_train_num回回す
            plsa.train(k=plsa_train_num)
            estimated_p_wd = np.dot(np.dot(plsa.Px_z.T, np.diag(plsa.Pz)), plsa.Py_z)
            with open(estimated_pwd_path, mode="a") as f:
                f.write(str(i + 1) + "回目\n")
                f.write("p(w|z)\n")
                for j in plsa.Px_z:
                    j_str: str = list(map(str, j))
                    f.write(" ".join(j_str))
                    f.write("\n")
                f.write("\n\n")

                f.write("estimated_p(w,d)\n")
                for j in estimated_p_wd:
                    j_str: str = list(map(str, j))
                    f.write(" ".join(j_str))
                    f.write("\n")
                f.write("\n\n")

            # 真のp(w, d)と計算したp(w, d)を求める
            kl = kl_diver(true_pwd, estimated_p_wd)
            with open(tmp_kl_path, mode="a") as f:
                f.write(str(i + 1) + "回目")
                f.write(str(kl))
                f.write("\n")

            # 作成したpz px_z py_z をたす
            tmp_kl += kl

        tmp_kl /= estimate_num


        '''
        生成したtmp_pz tmp_pw tmp_pdを保存する
        '''
        save_path_kl: str = os.getcwd() + "/data/kl/true_pwd/z_" + str(plsa_z_size) + ".txt"
        tmp_kl_str: str = str(tmp_kl)
        with open(save_path_kl, mode="a") as f:
            f.write("\n")
            f.write(tmp_kl_str)

