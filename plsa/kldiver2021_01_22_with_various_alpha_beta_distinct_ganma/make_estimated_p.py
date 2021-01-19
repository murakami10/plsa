import sys, os
sys.path.append('../')

import numpy as np
from typing import List

from used.CreatPWithParameter import CreatePWithParameter
from used.pre_process import zentai
from used.Plsa import Plsa

if __name__ == "__main__":

    # plsaにおけるトピックの数
    plsa_z_size = 15
    # plsaにおけるemステップの回数
    plsa_train_num = 100000
    # 作成するplsaの数
    estimate_num = 30

    # alpha beta gannmaを取得
    path = os.getcwd() + "/parameter_pd_with_various_beta.txt"
    p = CreatePWithParameter(path)

    # tmp_pzを初期化
    tmp_pz: np.ndarray = np.zeros(plsa_z_size)
    tmp_pw: np.ndarray = np.zeros((plsa_z_size, p.phi.shape[1]))
    tmp_pd: np.ndarray = np.zeros((plsa_z_size, p.Pd.shape[0]))
    # print(tmp_pz.shape)
    # print(tmp_pw.shape)
    # print(tmp_pd.shape)

    for i in range(estimate_num):
        print(str(i + 1) + "回目")
        # トピック数をplsa_z_sizeとしてplsaを実行
        p_wd = p.make_p(p.Pd.shape[0])
        # p(w, d)が全体で1になるように
        p_wd = zentai(p_wd)

        plsa = Plsa(p_wd, plsa_z_size)
        # plsa_train_num回回す
        plsa.train(k=plsa_train_num)
        # pz pw pd をソートする
        plsa.sort_pz_px_py()

        # 作成したpz px_z py_z をたす
        tmp_pz += plsa.Pz
        tmp_pw += plsa.Px_z
        tmp_pd += plsa.Py_z

    tmp_pz /= estimate_num
    tmp_pw /= estimate_num
    tmp_pd /= estimate_num


    '''
    生成したtmp_pz tmp_pw tmp_pdを保存する
    '''
    save_path_pz: str = os.getcwd() + "/data/pz/estimated_pz_z" + str(plsa_z_size) + ".txt"
    tmp_pz_str: List[str] = list(map(str, tmp_pz))
    with open(save_path_pz, mode="w") as f:
        f.write(' '.join(tmp_pz_str))

    save_path_pw: str = os.getcwd() + "/data/pw/estimated_pw_z" + str(plsa_z_size) + ".txt"
    with open(save_path_pw, mode="w") as f:
        for tmp in tmp_pw:
            tmp_pw_str: List[str] = list(map(str, tmp))
            f.write(' '.join(tmp_pw_str))
            f.write('\n')
        f.write('end')

    save_path_pd: str = os.getcwd() + "/data/pd/estimated_pd_z" + str(plsa_z_size) + ".txt"
    with open(save_path_pd, mode="w") as f:
        for tmp in tmp_pd:
            tmp_pd_str: List[str] = list(map(str, tmp))
            f.write(' '.join(tmp_pd_str))
            f.write('\n')
        f.write('end')

