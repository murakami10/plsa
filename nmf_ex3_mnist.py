#https://qiita.com/mine820/items/3d0c261192d51b87d95e
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import NMF

import keras
from keras.datasets import mnist

#訓練用とテスト用
(x_train, train_labels), (_, _) = mnist.load_data()

#print(x_train.shape)
# (60000, 28, 28) 60000 * 28 * 28のデータ

#画像を表示
#https://weblabo.oscasierra.net/python/keras-mnist-sample.html
#for i in range(0,1):
#    print("ラベル", train_labels[i])
#    # reshape(28, 28) 28 * 28の行列に変換する https://note.nkmk.me/python-numpy-reshape-usage/
#    # cmap https://beiznotes.org/matplot-cmap-list/
#    plt.imshow(x_train[i].reshape(28, 28), cmap='Blues')
#    plt.show()

x_train = x_train.reshape(-1, 784) # 2次元配列を1次元に変換 
#reshape(-1, x)にすると 列はxになり、行は残りから推測される https://qiita.com/yosshi4486/items/deb49d5a433a2c8a8ed4
#x_train = x_train.reshape(60000, 784)と同じ

#print(x_train.shape)
#(60000, 784)

#print(x_train.dtype) #uint8
x_train = x_train.astype('float32')   # int型をfloat32型に変換
#print(x_train.dtype) #float32

#print(x_train[0])
x_train /= 255                        # [0-255]の値を[0.0-1.0]に変換
#print(x_train[0])
print(x_train.shape[0], 'train samples') #(60000, 784)
print(x_train.shape) #(60000, 784)

x_train0 = np.array([]) 
for i in range(len(train_labels)):
    if train_labels[i] == 0:                 #ゼロというラベルのみ################################
        if x_train0.size == 0:
            x_train0 = np.array([x_train[i]])
        x_train0 = np.append(x_train0, [x_train[i]], axis=0)

x_train_T = x_train0.T #転置する
#print(x_train_T.shape) #np.shape はプロパティ

# 分解数
n_components = 1
#n_components = 10

#https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html 
#initで初期値の設定方法を指定できます(今回はランダム) よくわからん
model = NMF(n_components=n_components, init='random', random_state=0) # n_componentsで特徴の次元を指定

W = model.fit_transform(x_train_T) # 学習
H = model.components_

#print(W.shape) (784, 4)
#print(H.shape) (4, 60000)

# 28x28の画像に変換
W_image = W.T.reshape(n_components,28, 28)

# 画像の表示
fig = plt.figure(figsize=(5, 5)) #画像の大きさ(横幅、縦幅)
for i in range(0, n_components):
    #figure内の枠の大きさとどこに配置している。subplot(行の数,列の数,何番目に配置しているか)
    ax = fig.add_subplot(1, n_components, i+1)
    ax.imshow(W_image[i])

plt.show()
