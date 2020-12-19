import numpy as np

class PLSA:

    def __init__(self, N, Z):
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

        print()
        print('Pz')
        print(self.Pz)
        print('Px')
        print(self.Px_z)
        print('Py')
        print(self.Py_z)
        print()

    def train(self, k=200, t=1.0e-7):
        '''
        対数尤度が収束するまでEステMステを繰り返す(最大でk回)
        '''
        prev_llh = 100000
        for i in range(k):
            self.e_step()
            self.m_step()
            llh = self.llh()

            if abs((llh - prev_llh) / prev_llh) < t:
                break

            prev_llh = llh

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
        #print('################')
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

    def llh(self):
        '''
        対数尤度
        '''
        Pxy = self.Pz[None, None, :] \
            * self.Px_z.T[:, None, :] \
            * self.Py_z.T[None, :, :]
        Pxy = np.sum(Pxy, axis=2)
        Pxy /= np.sum(Pxy)

        return np.sum(self.N * np.log(Pxy))
