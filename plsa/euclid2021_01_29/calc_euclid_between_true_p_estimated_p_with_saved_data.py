import sys, os
sys.path.append('../')

import numpy as np
from typing import List

from used.CreatPWithParameter import CreatePWithParameter
from used.pre_process import zentai, euclid, euclid_ver2
from used.Plsa import Plsa



if __name__ == "__main__":

    plsazzz = [5, 7, 10, 13, 15]
    for zzz in plsazzz:
        # plsaにおけるトピックの数
        plsa_z_size = zzz
        print(plsa_z_size)
        # 作成するplsaの数
        estimate_num = 30

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

        # tmp_euclidを初期化
        tmp_euclid: float = 0
        tmp_euclid_v2: float = 0

        tmp_euclid_path: str = os.getcwd() + "/data/tmp_euclid/z_" + str(plsa_z_size) + ".txt"
        tmp_euclid_v2_path: str = os.getcwd() + "/data/tmp_euclid_v2/z_" + str(plsa_z_size) + ".txt"

        estimated_pwd_path: str = os.getcwd() + "/data/estimated_pwd/z_" + str(plsa_z_size) + ".txt"
        for i in range(estimate_num):
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
            eu_v1 = euclid(true_pwd, estimated_p_wd)
            eu_v2 = euclid_ver2(true_pwd, estimated_p_wd)

            with open(tmp_euclid_path, mode="a") as f:
                f.write(str(i + 1) + "回目")
                f.write(str(eu_v1))
                f.write("\n")

            with open(tmp_euclid_v2_path, mode="a") as f:
                f.write(str(i + 1) + "回目")
                f.write(str(eu_v2))
                f.write("\n")


            # 作成したpz px_z py_z をたす
            tmp_euclid += eu_v1
            tmp_euclid_v2 += eu_v2

        tmp_euclid /= estimate_num
        tmp_euclid_v2 /= estimate_num

        save_path_euclid: str = os.getcwd() + "/data/euclid/z_" + str(plsa_z_size) + ".txt"
        tmp_euclid_str: str = str(tmp_euclid)
        with open(save_path_euclid, mode="a") as f:
            f.write("\n")
            f.write(tmp_euclid_str)

        save_path_euclid_v2: str = os.getcwd() + "/data/euclid_v2/z_" + str(plsa_z_size) + ".txt"
        tmp_euclid_v2_str: str = str(tmp_euclid_v2)
        with open(save_path_euclid_v2, mode="a") as f:
            f.write("\n")
            f.write(tmp_euclid_v2_str)


