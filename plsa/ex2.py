
from used.plsa import PLSA
import cv2
import numpy as np
import matplotlib.pyplot as plt
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
Y = Y.reshape(4096, -1)
plsa = PLSA(Y, 20)
plsa.train()
z = plsa.Pz
x = plsa.Px_z
W = np.dot(x.T, np.diag(z))
H = plsa.Py_z


'''
nmf2 = NMF(n_components=1)
W = nmf2.fit_transform(Y) # 学習
H = nmf2.components_
'''

print(W.shape, H.shape)
#print(H.shape)

'''
fig, axes = plt.subplots(3,3,figsize=(10,10))
for i,(component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
    ax.imshow(component.reshape((64,64,3)))
    ax.set_title('component'+str(i+1))

plt.show()
'''

fig, axes = plt.subplots(4, 5, figsize=(8, 8))
for i, (component, ax) in enumerate(zip(W.T, axes.ravel())):
    ax.imshow(component.reshape((64, 64)), cmap='Greys')
    ax.set_title('c'+str(i+1))

plt.show()
'''
plt.imshow(W.T.reshape(64, 64), cmap='Greys')
'''

im = np.dot(W, H)
#print(im.shape)
im = im.T
#print(im.shape, im[0].shape)
#print(im[0].reshape(64, 64).shape)

plt.imshow(im[0].reshape(64, 64), cmap='Greys')
plt.show()

#7分