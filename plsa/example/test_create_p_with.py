import sys, os
sys.path.append('../')

import numpy as np
from typing import List

from used.CreatPWithParameter import CreatePWithParameter
from used.pre_process import zentai
from used.Plsa import Plsa

if __name__ == "__main__":


    # alpha beta gannmaを取得
    path = os.getcwd() + "/parameter_pd_with_various_beta.txt"
    p = CreatePWithParameter(path)
    pwd = p.make_p(p.Pd.shape[0])

    path = "./test.txt"

    with open(path, mode="w") as f:
        for p in pwd:
            p_str = list(map(str, p))
            f.write(" ".join(p_str))
            f.write("\n")
