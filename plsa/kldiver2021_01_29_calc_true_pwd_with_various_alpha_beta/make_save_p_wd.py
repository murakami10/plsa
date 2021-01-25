import sys, os
sys.path.append('../')

import numpy as np
from typing import List

from used.CreatPWithParameter import CreatePWithParameter

if __name__ == "__main__":

    plsazzz = [7, 10, 13, 15]
    for zzz in plsazzz:
        # plsaにおけるトピックの数
        plsa_z_size = zzz
        print(plsa_z_size)
        # 作成するplsaの数
        estimate_num = 30

        # alpha beta gannmaを取得
        path = os.getcwd() + "/parameter_pd_with_various_alpha_beta.txt"
        p = CreatePWithParameter(path)


        estimated_pwd_path: str = os.getcwd() + "/make_p_wd/" + str(plsa_z_size) + ".txt"
        for i in range(estimate_num):
            print(str(i + 1) + "回目")
            while True:
                # トピック数をplsa_z_size, １文章における単語作成数をw_size * 10としてplsaを実行
                p_wd, label = p.make_p_by_different_ways(p.Pd.shape[0], n_number=(p.w_size * 50))
                if not np.any(p_wd == 0):
                    break

            # p(w, d)が全体で1になるように
            with open(estimated_pwd_path, mode="a") as f:
                f.write(str(i + 1) + "回目\n")
                f.write("make_p(w, d)\n")
                f.write(" ".join([str(j) for j in label]))
                f.write("\n")
                for j in p_wd:
                    j_str: str = [str(jj) for jj in j]
                    f.write(" ".join(j_str))
                    f.write("\n")
                f.write("\n\n")
