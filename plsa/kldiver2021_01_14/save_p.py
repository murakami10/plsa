import numpy as np
import sys, os
from typing import List
sys.path.append('../')

from used.CreatP import CreatP

if __name__ == "__main__":

    path = os.getcwd() + "/parameter.txt"
    try:
        with open(path) as f:
            while True:
                line = (f.readline()).strip()
                if line == "alpha":
                    alpha = list(map(float, (f.readline()).split()))
                    alpha = np.array(alpha)

                if line == "beta":
                    beta = list(map(float, (f.readline()).split()))
                    beta = np.array(beta)

                if line == "ganma":
                    ganma = list(map(float, (f.readline()).split()))
                    ganma = np.array(ganma)
                    break
    except FileNotFoundError:
        print("file not found")
        exit()

    p = CreatP(alpha, beta, ganma)

    print(p.Pd.shape)
    print()
    print(p.Pz_d.shape)
    print()
    print(p.phi.shape)

    path = os.getcwd() + "/parameter_pd.txt"
    try:
        with open(path, mode="x") as f:
            f.write('pd \n')
            pd: List[int] = list(map(str, list(p.Pd)))
            f.write(' '.join(pd))
            f.write('\n')
            f.write('\n')

            f.write('pz_d \n')
            for pz_d in p.Pz_d:
                pz: List[int] = list(map(str, list(pz_d)))
                f.write(' '.join(pz))
                f.write('\n')

            f.write('end\n')
            f.write('\n')

            f.write('phi \n')
            for phi in p.phi:
                ph: List[int] = list(map(str, list(phi)))
                f.write(' '.join(ph))
                f.write('\n')

            f.write('end\n')
    except(FileExistsError):
        print('file exit')
        exit()

