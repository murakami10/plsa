import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import copy

sample_num = 1

#色(plane)の並びの扱いが異なるから
#OpenCVはBGR順、matplotlibはRGB順を前提として扱う
#cv2.cvtColor(img, cv2.COLOR_BGR2RGB)にてRGB順に並べ替えることでmatplotlibでも正しい色で表示できるようになります

X = []
Y = []
sample_num +=1

for i in range(1, sample_num):
    filename = './nmf_ex4/' + str(i) + '.png'
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)
    img = img.reshape(64*64)/255
    X.append(img)


X = np.array(X)
print(X)
plt.imshow(X.reshape(64, 64), cmap='Greys')
#plt.imshow(X.reshape(64, 64),vmin=0, vmax=255, cmap='Greys') #カラーバーを設定 これは正規化してないときに使う
plt.show()