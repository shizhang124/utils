#coding:utf-8

from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn import cluster
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import manifold


def k_mean(data, save_path = "km_model.m"):
    estimator = KMeans(n_clusters=6, n_jobs=-1)
    res = estimator.fit_predict(data)

    lable_pred = estimator.labels_
    centroids = estimator.cluster_centers_
    inertia = estimator.inertia_

    print((time.time() - s_time) / 1000.0, 's')
    joblib.dump(estimator, save_path)

    return lable_pred


def t_SNE(X, Y, save_path='t-SNE.png'):

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)

    print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(15, 15))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(Y[i]), color=plt.cm.Set1(Y[i]),
                 fontdict={'weight': 'bold', 'size': 8})
    plt.xticks([])
    plt.yticks([])
    plt.savefig("t-SNE.png")
    # plt.show()


if __name__ == '__main__':
    s_time = time.time()
    data = np.random.rand(100, 20)
    print(data.shape)
    lable_pred = k_mean(data)

    t_SNE(data, lable_pred)




    '''
    sklearn.cluster.KMeans(
        n_clusters=8,
        init='k-means++', 
        n_init=10, 
        max_iter=300, 
        tol=0.0001, 
        precompute_distances='auto', 
        verbose=0, 
        random_state=None, 
        copy_x=True, 
        n_jobs=1, 
        algorithm='auto'
        )
    n_clusters: 簇的个数，即你想聚成几类
    init: 初始簇中心的获取方法
    n_init: 获取初始簇中心的更迭次数，为了弥补初始质心的影响，算法默认会初始10个质心，实现算法，然后返回最好的结果。
    max_iter: 最大迭代次数（因为kmeans算法的实现需要迭代）
    tol: 容忍度，即kmeans运行准则收敛的条件
    precompute_distances:是否需要提前计算距离，这个参数会在空间和时间之间做权衡，如果是True 会把整个距离矩阵都放到内存中，auto 会默认在数据样本大于featurs*samples 的数量大于12e6 的时候False,False 时核心实现的方法是利用Cpython 来实现的
    verbose: 冗长模式（不太懂是啥意思，反正一般不去改默认值）
    random_state: 随机生成簇中心的状态条件。
    copy_x: 对是否修改数据的一个标记，如果True，即复制了就不会修改数据。bool 在scikit-learn 很多接口中都会有这个参数的，就是是否对输入数据继续copy 操作，以便不修改用户的输入数据。这个要理解Python 的内存机制才会比较清楚。
    n_jobs: 并行设置
    algorithm: kmeans的实现算法，有：’auto’, ‘full’, ‘elkan’, 其中 ‘full’表示用EM方式实现
    虽然有很多参数，但是都已经给出了默认值。所以我们一般不需要去传入这些参数,参数的。可以根据实际需要来调用。
    '''