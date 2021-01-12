from typing import List
import numpy as np

from used.CreatP import CreatP


class CreatePWithParameter(CreatP):

    def __init__(self, file_path: str):

        with open(file_path) as f:
            while True:
                line = (f.readline()).strip()

                if line == 'pd':
                    self.Pd = list(map(float, f.readline().split()))
                    self.Pd = np.array(self.Pd)

                if line == 'pz_d':
                    self.Pz_d: List[int] = []
                    while True:
                        line = (f.readline()).strip()
                        if line == 'end':
                            break
                        self.Pz_d.append(list(map(float, line.split())))
                    self.Pz_d = np.array(self.Pz_d)

                if line == 'phi':
                    self.phi: List[int] = []
                    while True:
                        line = (f.readline()).strip()
                        if line == 'end':
                            break
                        self.phi.append(list(map(float, line.split())))

                    self.phi = np.array(self.phi)
                    break
        self.w_size = self.phi.shape[1]
        self.z_size = self.phi.shape[0]
