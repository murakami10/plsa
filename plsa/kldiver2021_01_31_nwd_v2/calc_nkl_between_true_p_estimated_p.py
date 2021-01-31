import sys, os
sys.path.append('../')

import numpy as np
from typing import List

from used.CreatPWithParameter import CreatePWithParameter

def nkl_diver_with_label(A: np.ndarray, B: np.ndarray, lable: List[int], create_p: CreatePWithParameter, esnum: int, plsa_size: int) -> (float, List[float]):
    tmp_AnCn = 0
    cycle = 10
    path = os.getcwd() + "/data/tmp_nkl_matrix/" + str(plsa_size) + ".txt"
    with open(path, mode="a") as f:
        f.write(str(esnum + 1) + "回目\n")
        f.write("\n\n")

    cycle_list:List[float] = []

    C: np.ndarray = np.zeros_like(B)
    for index, l in enumerate(lable):
        C[:, l] += B[:, index]

    A_tmp = np.ma.log(A)
    C_tmp = np.ma.log(C)
    C_tmp = C_tmp.filled(fill_value=0.0) # マスクされている部分をゼロに変換

    for i in range(cycle):
        print("cycle: " + str(i))

        n: np.ndarray = np.zeros_like(B)
        label_set = set(label)
        for l in label_set:
            while True:
                n_wd = create_p.make_w(l, n_number=50)
                if not np.any(n_wd == 0):
                    n_wd: np.ndarray = np.array(n_wd)
                    break
            n[:, l] += n_wd.T

        An: np.ndarray = n * A_tmp
        Cn: np.ndarray = n * C_tmp


        with open(path, mode="a") as f:
            f.write(str(i + 1) + "cycle\n")

            f.write("n\n")
            for a in n:
                a_str = list(map(str, a))
                f.write(" ".join(a_str))
                f.write("\n")
            f.write("\n\n")

            f.write("An\n")
            for a in An:
                a_str = list(map(str, a))
                f.write(" ".join(a_str))
                f.write("\n")
            f.write("\n\n")

            f.write("Cn\n")
            for a in Cn:
                a_str = list(map(str, a))
                f.write(" ".join(a_str))
                f.write("\n")
            f.write("\n\n")

        An = An.sum()
        Cn = Cn.sum()

        AnCn_sum = An / Cn

        cycle_list.append(AnCn_sum)
        tmp_AnCn += AnCn_sum
    tmp_nab = tmp_AnCn/cycle
    return tmp_nab, cycle_list


if __name__ == "__main__":

    plsazzz = [5,7,8,9,10,11,12,13,15]
    for zzz in plsazzz:
        # plsaにおけるトピックの数
        plsa_z_size = zzz
        print(plsa_z_size)
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

        # tmp_klを初期化
        tmp_nkl: float = 0
        tmp_nkl_path: str = os.getcwd() + "/data/tmp_nkl/z_" + str(plsa_z_size) + ".txt"
        estimated_pwd_path: str = os.getcwd() + "/data/estimated_pwd/z_" + str(plsa_z_size) + ".txt"
        make_p_wd: str = os.getcwd() + "/make_p_wd/" + str(plsa_z_size) + ".txt"
        each_result = []
        for i in range(estimate_num):
            # lableを取得
            with open(make_p_wd) as f:
                while True:
                    line = f.readline().strip()
                    if line == str(i+1) + "回目":
                        line = f.readline().strip() # p(w, d)が入力される
                        break
                label: List[int] = [int(j) for j in f.readline().strip().split()]

            # estimated_p(w,d)を取得
            with open(estimated_pwd_path) as f:
                while True:
                    line = f.readline().strip()
                    if line == str(i+1) + "回目":
                        line = f.readline().strip() # p(w, d)が入力される
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
                    estimated_p_wd.append([float(x) for x in line.split()])
                estimated_p_wd: np.ndarray = np.array(estimated_p_wd)

            path_create_p = os.getcwd() + "/parameter_pd_with_various_alpha_beta.txt"
            create_p = CreatePWithParameter(path_create_p)

            print(str(i + 1) + "回目")

            # 真のp(w, d)と計算したp(w, d)を求める
            nkl, cycle_list = nkl_diver_with_label(true_pwd, estimated_p_wd, label, create_p, i, plsa_z_size)

            each_result.append(cycle_list)

            with open(tmp_nkl_path, mode="a") as f:
                f.write(str(i + 1) + "回目")
                f.write(str(nkl))
                f.write("\n")

            # 作成したpz px_z py_z をたす
            tmp_nkl += nkl

        tmp_nkl /= estimate_num

        with open(tmp_nkl_path, mode="a") as f:
            f.write("\n\n")
            f.write("each_result\n")
            for er in each_result:
                er_str = list(map(str, er))
                f.write(" ".join(er_str))
                f.write("\n")
            f.write("\n\n")

        save_path_nkl: str = os.getcwd() + "/data/nkl/z_" + str(plsa_z_size) + ".txt"
        tmp_nkl_str: str = str(tmp_nkl)
        with open(save_path_nkl, mode="a") as f:
            f.write(tmp_nkl_str)
            f.write("\n")

