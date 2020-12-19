import cv2
import numpy as np
import copy

sample_num = 1000
#sample_num = 1

#色(plane)の並びの扱いが異なるから
#OpenCVはBGR順、matplotlibはRGB順を前提として扱う
#cv2.cvtColor(img, cv2.COLOR_BGR2RGB)にてRGB順に並べ替えることでmatplotlibでも正しい色で表示できるようになります

X = []
Y = []
sample_num +=1

for i in range(1, sample_num):
    filename = '../nmf/nmf_ex4/' + str(i) + '.png'
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)
    img = img.reshape(64*64)
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
print(Y.shape)