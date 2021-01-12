import sys, os
sys.path.append('../')

from used.CreatPWithParameter import CreatePWithParameter
from used.pre_process import zentai
from used.Plsa import Plsa

if __name__ == "__main__":

    path = os.getcwd() + "/parameter_pd.txt"
    p = CreatePWithParameter(path)
    p_wd = p.make_p(p.Pd.shape[0])
    p_wd = zentai(p_wd)

    plsa = Plsa()

