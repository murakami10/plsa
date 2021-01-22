import sys, os
sys.path.append('../')

import numpy as np
from typing import List

from used.CreatPWithParameter import CreatePWithParameter
from used.pre_process import zentai, kl_diver
from used.Plsa import Plsa


def kl_diver_save(A: np.ndarray, B: np.ndarray, plsa_z_size,num) -> float:

    tmp_kl_path: str = os.getcwd() + "/data/tmp_kl/from_saved_data/ab/z_" + str(plsa_z_size) + ".txt"
    ab = np.divide(A, B, out=np.zeros_like(A), where=(B != 0))

    with open(tmp_kl_path, mode='a') as f:
        f.write(str(num) + "\n")
        f.write("ab \n")
        for i in ab:
            ab_str = [str(j) for j in i]
            f.write(" ".join(ab_str))
            f.write("\n")
        f.write("\n")
    ab = np.ma.log(ab)
    with open(tmp_kl_path, mode='a') as f:
        f.write("log(ab) \n")
        for i in ab:
            ab_str = [str(j) for j in i]
            f.write(" ".join(ab_str))
            f.write("\n")
        f.write("\n")
    ab = ab.sum()

    return ab


if __name__ == "__main__":

    plsazzz = [5, 7, 10, 13, 15]
    for zzz in plsazzz:
        # plsaにおけるトピックの数
        plsa_z_size = zzz
        print(plsa_z_size)
        # 作成するplsaの数
        estimate_num = 1

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
        tmp_kl_path: str = os.getcwd() + "/data/tmp_kl/from_saved_data/z_" + str(plsa_z_size) + ".txt"
        estimated_pwd_path: str = os.getcwd() + "/data/estimated_pwd/z_" + str(plsa_z_size) + ".txt"
        estimate_num_list = [20,12]
        for i in estimate_num_list:
            print(str(i + 1) + "回目")

            with open(estimated_pwd_path) as f:
                while True:
                    line = f.readline().strip()
                    if line == str(i+1) + "回目":
                        break

                while True:
                    line = f.readline().strip()
                    if line == "estimated_p(w,d)":
                        break

                estimated_p_wd = []
                while True:
                    line = f.readline().strip()
                    if line == "":
                        break
                    estimated_p_wd.append(list(map(float, line.split())))
                estimated_p_wd: np.ndarray = np.array(estimated_p_wd)

            # 真のp(w, d)と計算したp(w, d)を求める
            kl = kl_diver_save(true_pwd, estimated_p_wd, plsa_z_size, i+1)
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
        save_path_kl: str = os.getcwd() + "/data/kl/true_pwd/from_saved_data/z_" + str(plsa_z_size) + ".txt"
        tmp_kl_str: str = str(tmp_kl)
        with open(save_path_kl, mode="a") as f:
            f.write("\n")
            f.write(tmp_kl_str)

