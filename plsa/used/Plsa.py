import numpy as np

class Plsa:

    def __init__(self, N: np.ndarray, Z: int):
        self.N = N
        self.X = N.shape[0]
        self.Y = N.shape[1]
        self.Z = Z

        # P(z) 初期化
        self.Pz = np.random.rand(self.Z)
        # P(x|z)
        self.Px_z = np.random.rand(self.Z, self.X)
        # P(y|z)
        self.Py_z = np.random.rand(self.Z, self.Y)

        #正規化
        self.Pz = self.Pz / np.sum(self.Pz)
        self.Px_z = self.Px_z / np.sum(self.Px_z, axis=1)[:, None]
        self.Py_z = self.Py_z / np.sum(self.Py_z, axis=1)[:, None]

        '''
        print()
        print('Pz')
        print(self.Pz)
        print('Px')
        print(self.Px_z)
        print('Py')
        print(self.Py_z)
        print()
        '''

    def train(self, k: int = 200):
        '''
        EMステップをk回繰り返す
        '''
        for i in range(k):
            self.e_step()
            self.m_step()

    def e_step(self):
        # ここのNoneは新しい次元を追加する https://note.nkmk.me/python-numpy-newaxis/
        #print(self.Pz[None, None, :])
        #print(self.Pz[None, None, :].shape)
        #print(self.Px_z.T[:, None, :])
        #print(self.Px_z.T[:, None, :].shape)
        #print(self.Py_z.T[None, :, :])
        #print(self.Py_z.T[None, :, :].shape)

        #ここの積はアダマール積
        self.Pz_xy = self.Pz[None, None, :] * self.Px_z.T[:, None, :] * self.Py_z.T[None, :, :]
        #print(self.Pz_xy)
        #print(self.Pz_xy.shape)
        d = np.sum(self.Pz_xy, axis=2)[:, :, None]
        self.Pz_xy = np.divide(self.Pz_xy, d, out=np.zeros_like(self.Pz_xy), where=(d != 0))
        #print(self.Pz_xy)

    def m_step(self):

        NP = self.N[:, :, None] * self.Pz_xy
        #print(self.Pz_xy)
        #print(NP)

        #print(NP)
        self.Pz = np.sum(NP, axis=(0, 1))
        #print(self.Pz)
        self.Px_z = np.sum(NP, axis=1).T
        self.Py_z = np.sum(NP, axis=0).T

        self.Pz /= np.sum(self.Pz)
        #print(self.Pz)
        self.Px_z /= np.sum(self.Px_z, axis=1)[:, None]
        self.Py_z /= np.sum(self.Py_z, axis=1)[:, None]


    def sort_pz_px_py(self):
        # pzを小さい順に並べ、その並べ順にPx_z Py_zも並べる

        index_sort = np.argsort(self.Pz)
        self.Pz = self.Pz[index_sort]
        self.Px_z = self.Px_z[index_sort, :]
        self.Py_z = self.Py_z[index_sort, :]
