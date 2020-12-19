#https://qiita.com/hibit/items/8f0525ab1b616061c630

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import copy

sample_num = 1000


X = []
Y = []
sample_num += 1

for i in range(1, sample_num):
    filename = './nmf_ex4/' + str(i) + '.png'
    img = cv2.imread(filename)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)
    img = img.reshape(64*64)/255
    X.append(img)

#画像1つの要素は行（高さ） x 列（幅） x 色（3）の三次元のndarrayとなる https://note.nkmk.me/python-opencv-imread-imwrite/
#print(img.shape)
Y = copy.deepcopy(X)
'''
X = np.array(X)
nmf = NMF(n_components=15)
nmf.fit(X)
X_nmf = nmf.transform(X)
print(X_nmf.shape, nmf.components_.shape)
'''

Y = np.array(Y)
Y = Y.T
print(Y.shape)
Y = Y.reshape(4096, -1)
nmf2 = NMF(n_components=90)
W = nmf2.fit_transform(Y) # 学習
H = nmf2.components_

print(W.shape, H.shape)
#print(H.shape)

im = np.dot(W, H)
#print(im.shape)
im = im.T
#print(im.shape, im[0].shape)
print(im[0].reshape(64, 64).shape)

plt.imshow(im[0].reshape(64, 64),cmap='Greys')
plt.show()

