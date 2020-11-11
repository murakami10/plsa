#https://qiita.com/nozma/items/d8dafe4e938c43fb7ad1
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

X = [
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1]
    ]

nmf = NMF(n_components=3) #X = W :6x3 H :3x6
nmf.fit(X)

W = nmf.fit_transform(X)
print(np.round(W, 1)) #少数第2を四捨五入
print('#########################')
'''
[[0.  1.2 0. ]
 [0.  1.2 0. ]
 [0.9 0.  0. ]
 [0.9 0.  0. ]
 [0.  0.  0.8]
 [0.  0.  0.8]]
 '''

H = nmf.components_
print(np.round(H, 1))
print('#########################')
'''
[[0.  0.  1.1 1.1 1.1 0. ]
 [0.9 0.9 0.9 0.  0.  0. ]
 [0.  0.  0.  1.3 1.3 1.3]]
 '''

WH = np.dot(W, H) #行列の掛け算
print(np.round(WH, 1))
print('#########################')
'''
[[1. 1. 1. 0. 0. 0.]
 [1. 1. 1. 0. 0. 0.]
 [0. 0. 1. 1. 1. 0.]
 [0. 0. 1. 1. 1. 0.]
 [0. 0. 0. 1. 1. 1.]
 [0. 0. 0. 1. 1. 1.]]
'''
