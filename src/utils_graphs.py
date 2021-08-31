import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from fitter import Fitter
import pandas as pd
from scipy.spatial import distance
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors

# # test data
# m, n = 50, 10
# X = 10.0 * np.random.random_sample((m, n))
# vectori, vectorj = X[0, :], X[1, :]

def get_pos_xy(X):
    (m, n) = X.shape
    if n < 2:
        return None
    pos = {i: np.array([X[i, 0], X[i, 1]]) for i in range(m)}
    return pos

# PlotGraph
def simPlotGraph(G, with_labels=True):
    pos = nx.kamada_kawai_layout(G)
    nodecolor = G.degree(weight='weight')  # 度数越大，节点越大，连接边颜色越深
    nodecolor2 = pd.DataFrame(nodecolor)  # 转化称矩阵形式
    nodecolor3 = nodecolor2.iloc[:, 1]  # 索引第二列
    edgecolor = range(G.number_of_edges())  # 设置边权颜色
    nx.draw(G,
            pos,
            with_labels=with_labels,
            node_size=200,
            # node_size=nodecolor3 * 6 * 10,# 度数越大，节点越大
            node_color=nodecolor3 * 5,
            edge_color=edgecolor)
    # nx.draw(G, pos=pos, node_size=200, with_labels=True, node_color='red')
    plt.show()

# PlotGraph
def simPlotGraph_XY(G, X, with_labels=True):
    pos = get_pos_xy(X)
    nodecolor = G.degree(weight='weight')  # 度数越大，节点越大，连接边颜色越深
    nodecolor2 = pd.DataFrame(nodecolor)  # 转化称矩阵形式
    nodecolor3 = nodecolor2.iloc[:, 1]  # 索引第二列
    edgecolor = range(G.number_of_edges())  # 设置边权颜色
    nx.draw(G,
            pos,
            with_labels=with_labels,
            node_size=200,
            # node_size=nodecolor3 * 6 * 10,# 度数越大，节点越大
            node_color=nodecolor3 * 5,
            edge_color=edgecolor)
    # nx.draw(G, pos=pos, node_size=200, with_labels=True, node_color='red')
    plt.show()

# Plot Graph
def plot_graph(G, weight='weight', with_labels=False, save=False, filename='./graph.svg'):
    # test: outer point
    print("G: number_of_nodes:{0}, number_of_edges:{1}".format(G.number_of_nodes(), G.number_of_edges()))
    # pos = nx.spring_layout(G)  # set layout
    pos = nx.kamada_kawai_layout(G)
    # nx.draw(G, pos=pos, node_size=300, with_labels=with_labels, node_color='red')
    nodecolor = G.degree(weight=weight)  # 度数越大，节点越大，连接边颜色越深
    nodecolor2 = pd.DataFrame(nodecolor)  # 转化称矩阵形式
    nodecolor3 = nodecolor2.iloc[:, 1]  # 索引第二列
    edgecolor = range(G.number_of_edges())  # 设置边权颜色
    nx.draw(G,
            pos,
            with_labels=with_labels,
            node_size=nodecolor3 * 6 * 10,
            node_color=nodecolor3 * 5,
            edge_color=edgecolor)
    if save:
        plt.savefig(filename, dpi=600, transparent=True)
    plt.show()

def label_result(X, result):
    y_result = np.zeros((X.shape[0],)) - 1# -1 为噪声点
    cls = 0
    for p in result:
        y_result[p] = cls
        cls += 1
    return y_result

# dis distribution
def get_distributions(data, fitter=False, savefig=False, name_dis='', name_str=''):
    if fitter:
        # 利用fitter拟合数据样本的分布
        # may take some time since by default, all distributions are tried
        # but you call manually provide a smaller set of distributions
        f = Fitter(data, xmin=None, xmax=None, bins=100, distributions=['norm', 't', 'laplace'])
        f.fit()
        f.summary() #返回排序好的分布拟合质量（拟合效果从好到坏）,并绘制数据分布和Nbest分布
        f.hist() #绘制组数=bins的标准化直方图
        # f.plot_pdf(names=None, Nbest=3, lw=2) #绘制分布的概率密度函数
        print(f.summary())
    else:
        (n, bins) = np.histogram(data, bins=100, density=True)
        plt.plot(.5*(bins[1:] + bins[:-1]), n)
        plt.title("%s histogram" % name_str, size=14)
        if savefig:
            plt.savefig('./data/' + name_dis + '_' + name_str + '_histogram.png', bbox_inches='tight', dpi=300, transparent=True)
        plt.show()

# n, m = G.number_of_nodes(), G.number_of_edges()
# Eg_List = [(u, v, w['weight']) for (u, v, w) in G.edges(data=True)]
def create_weight_graph_txt_file(G, filename='InputGraph'):
    file = open(filename+'.txt', mode='w')
    str_number_of_nodes = '{0}\n'.format(G.number_of_nodes())
    file.write(str_number_of_nodes)
    for (u, v, w) in G.edges(data=True):
        str_edge = '{0} {1} {2}\n'.format(u, v, w['weight'])
        file.write(str_edge)
    file.close()


# # minkowski
# dis_max = np.max([distance.minkowski(X[i, :], X[j, :], p=3) for i in range(m) for j in range(m) if i != j])
# dis_canberra = distance.minkowski(vectori, vectorj, p=3)
def get_nxGraph_minkowski(X, epsw=1.0 / 5.0, p=3):
    (m, n) = X.shape
    edges_list = []
    dis_all = []
    dis_max = np.max([distance.minkowski(X[i, :], X[j, :], p=p) for i in range(m) for j in range(m) if i != j])
    for i in range(m):
        for j in range(m):
            if i != j:
                dis_0 = distance.minkowski(X[i, :], X[j, :], p=p)# p=2 isequivalent to euclidean
                dis_all.append(dis_0)
                if dis_0 <= 1e-5:
                    weight = dis_max
                else:
                    weight = 1.0 / dis_0
                if weight >= epsw:
                    edge = (i, j, {'weight': weight})
                    edges_list.append(edge)
    G = nx.Graph()
    G.add_edges_from(edges_list)
    
    
    
    return G, np.array(dis_all)
# G, dis_all = get_nxGraph_minkowski(X, epsw=1.0 / 12.0)
# simPlotGraph(G)
# get_distributions(dis_all)


# # canberra
# dis_max = np.max([distance.canberra(X[i, :], X[j, :]) for i in range(m) for j in range(n) if i != j])
# dis_canberra = distance.canberra(vectori, vectorj)
def get_nxGraph_canberra(X, epsw=1.0 / 5.0):
    (m, n) = X.shape
    edges_list = []
    dis_all = []
    dis_max = np.max([distance.canberra(X[i, :], X[j, :]) for i in range(m) for j in range(m) if i != j])
    for i in range(m):
        for j in range(m):
            if i != j:
                dis = distance.canberra(X[i, :], X[j, :])
                dis_all.append(dis)
                if dis <= 1e-5:
                    weight = dis_max
                else:
                    weight = 1.0 / dis
                if weight >= epsw:
                    edge = (i, j, {'weight': weight})
                    edges_list.append(edge)
    G = nx.Graph()
    G.add_edges_from(edges_list)
    return G, np.array(dis_all)
# G, dis_all = get_nxGraph_canberra(X, epsw=1.0 / 5.0)
# simPlotGraph(G)
# get_distributions(dis_all)

# # braycurtis
# dis_max = np.max([distance.braycurtis(X[i, :], X[j, :]) for i in range(m) for j in range(n) if i != j])
# dis_braycurtis = distance.braycurtis(vectori, vectorj)
def get_nxGraph_braycurtis(X, epsw=1.0 / 5.0):
    (m, n) = X.shape
    edges_list = []
    dis_all = []
    dis_max = np.max([distance.braycurtis(X[i, :], X[j, :]) for i in range(m) for j in range(m) if i != j])
    for i in range(m):
        for j in range(m):
            if i != j:
                dis = distance.braycurtis(X[i, :], X[j, :])
                dis_all.append(dis)
                if dis <= 1e-5:
                    weight = dis_max
                else:
                    weight = 1.0 / dis
                if weight >= epsw:
                    edge = (i, j, {'weight': weight})
                    edges_list.append(edge)
    G = nx.Graph()
    G.add_edges_from(edges_list)
    return G, np.array(dis_all)
# G, dis_all = get_nxGraph_braycurtis(X, epsw=1.0 / 0.55)
# simPlotGraph(G)
# get_distributions(dis_all)


# # chebyshev
# dis_max = np.max([distance.chebyshev(X[i, :], X[j, :]) for i in range(m) for j in range(n) if i != j])
# dis_chebyshev = distance.chebyshev(vectori, vectorj)
def get_nxGraph_chebyshev(X, epsw=1.0 / 5.0):
    (m, n) = X.shape
    edges_list = []
    dis_all = []
    dis_max = np.max([distance.chebyshev(X[i, :], X[j, :]) for i in range(m) for j in range(m) if i != j])
    for i in range(m):
        for j in range(m):
            if i != j:
                dis = distance.chebyshev(X[i, :], X[j, :])
                dis_all.append(dis)
                if dis <= 1e-5:
                    weight = dis_max
                else:
                    weight = 1.0 / dis
                if weight >= epsw:
                    edge = (i, j, {'weight': weight})
                    edges_list.append(edge)
    G = nx.Graph()
    G.add_edges_from(edges_list)
    return G, np.array(dis_all)
# G, dis_all = get_nxGraph_chebyshev(X, epsw=1.0 / 9.0)
# simPlotGraph(G)
# get_distributions(dis_all)

# # manhattan
# dis_max = np.max([distance.cityblock(X[i, :], X[j, :]) for i in range(m) for j in range(n) if i != j])
# dis_manhattan = distance.cityblock(vectori, vectorj)
def get_nxGraph_manhattan(X, epsw=1.0 / 5.0):
    (m, n) = X.shape
    edges_list = []
    dis_all = []
    dis_max = np.max([distance.cityblock(X[i, :], X[j, :]) for i in range(m) for j in range(m) if i != j])
    for i in range(m):
        for j in range(m):
            if i != j:
                dis = distance.cityblock(X[i, :], X[j, :])
                dis_all.append(dis)
                if dis <= 1e-5:
                    weight = dis_max
                else:
                    weight = 1.0 / dis
                if weight >= epsw:
                    edge = (i, j, {'weight': weight})
                    edges_list.append(edge)
    G = nx.Graph()
    G.add_edges_from(edges_list)
    return G, np.array(dis_all)
# G, dis_all = get_nxGraph_manhattan(X, epsw=1.0 / 50.0)
# simPlotGraph(G)
# get_distributions(dis_all)


# # correlation
# dis_max = np.max([distance.correlation(X[i, :], X[j, :]) for i in range(m) for j in range(m) if i != j])
# dis_correlation = distance.correlation(vectori, vectorj)
def get_nxGraph_correlation(X, epsw=0.5):
    (m, n) = X.shape
    edges_list = []
    dis_all = []
    # dis_max = np.max([distance.correlation(X[i, :], X[j, :]) for i in range(m) for j in range(m) if i != j])
    for i in range(m):
        for j in range(m):
            if i != j:
                dis = distance.correlation(X[i, :], X[j, :])
                dis_all.append(dis)
                weight = dis
                if weight >= epsw:
                    edge = (i, j, {'weight': weight})
                    edges_list.append(edge)
    G = nx.Graph()
    G.add_edges_from(edges_list)
    return G, np.array(dis_all)
# G, dis_all = get_nxGraph_correlation(X, epsw=0.5)
# simPlotGraph(G)
# get_distributions(dis_all)


# # pearsonr correlation
# dis_max = np.max([pearsonr(X[i, :], X[j, :]) for i in range(m) for j in range(m) if i != j])
# dis_pearsonr = np.max(pearsonr(vectori, vectorj))
def get_nxGraph_pearsonr(X, epsw=0.5):
    (m, n) = X.shape
    edges_list = []
    dis_all = []
    # dis_max = np.max([pearsonr(X[i, :], X[j, :]) for i in range(m) for j in range(m) if i != j])
    for i in range(m):
        for j in range(m):
            if i != j:
                dis = np.max(pearsonr(X[i, :], X[j, :]))
                dis_all.append(dis)
                weight = dis
                if weight >= epsw:
                    edge = (i, j, {'weight': weight})
                    edges_list.append(edge)
    G = nx.Graph()
    G.add_edges_from(edges_list)
    return G, np.array(dis_all)
# G, dis_all = get_nxGraph_pearsonr(X, epsw=0.5)
# simPlotGraph(G)
# get_distributions(dis_all)

# # cosine
# dis_max = np.max([distance.cosine(X[i, :], X[j, :]) for i in range(m) for j in range(m) if i != j])
# dis_cosine = distance.cosine(vectori, vectorj)
def get_nxGraph_cosine(X, epsw=0.5):
    (m, n) = X.shape
    edges_list = []
    dis_all = []
    # dis_max = np.max([distance.cosine(X[i, :], X[j, :]) for i in range(m) for j in range(m) if i != j])
    for i in range(m):
        for j in range(m):
            if i != j:
                dis = distance.cosine(X[i, :], X[j, :])
                dis_all.append(dis)
                weight = dis
                if weight >= epsw:
                    edge = (i, j, {'weight': weight})
                    edges_list.append(edge)
    G = nx.Graph()
    G.add_edges_from(edges_list)
    return G, np.array(dis_all)
# G, dis_all = get_nxGraph_cosine(X, epsw=0.25)
# simPlotGraph(G)
# get_distributions(dis_all)


# # euclidean
# dis_max = np.max([distance.euclidean(X[i, :], X[j, :]) for i in range(m) for j in range(m) if i != j])
# dis_euclidean = distance.euclidean(vectori, vectorj)
def get_nxGraph_euclidean(X, epsw=1.0 / 5.0):
    (m, n) = X.shape
    edges_list = []
    dis_all = []
    dis_max = np.max([distance.euclidean(X[i, :], X[j, :]) for i in range(m) for j in range(m) if i != j])
    for i in range(m):
        for j in range(m):
            if i != j:
                dis = distance.euclidean(X[i, :], X[j, :])
                dis_all.append(dis)
                if dis <= 1e-5:
                    weight = dis_max
                else:
                    weight = 1.0 / dis
                if weight >= epsw:
                    edge = (i, j, {'weight': weight})
                    edges_list.append(edge)
    G = nx.Graph()
    G.add_edges_from(edges_list)
    return G, np.array(dis_all)
# G, dis_all = get_nxGraph_euclidean(X, epsw=1.0 / 16.0)
# simPlotGraph(G)
# get_distributions(dis_all)


# # mahalanobis
# vectori_mahala, vectorj_mahala = vectori[:, np.newaxis], vectorj[:, np.newaxis]
# cov_X = np.cov(np.hstack((vectori_mahala, vectorj_mahala)))
# dis_mahalanobis = distance.mahalanobis(vectori, vectorj, VI=cov_X)
def get_nxGraph_mahalanobis(X, epsw=1.0 / 5.0):
    (m, n) = X.shape
    edges_list = []
    dis_all = []
    dis_max = []
    for i in range(m):
        for j in range(m):
            if i != j:
                vectori, vectorj = X[i, :], X[j, :]
                vectori_mahala, vectorj_mahala = vectori[:, np.newaxis], vectorj[:, np.newaxis]
                cov_X = np.cov(np.hstack((vectori_mahala, vectorj_mahala)))
                dis_mahalanobis = distance.mahalanobis(vectori, vectorj, VI=cov_X)
                dis_max.append(dis_mahalanobis)
                dis_all.append(dis_mahalanobis)
    dis_max = np.max(dis_max)
    for i in range(m):
        for j in range(m):
            if i != j:
                vectori, vectorj = X[i, :], X[j, :]
                vectori_mahala, vectorj_mahala = vectori[:, np.newaxis], vectorj[:, np.newaxis]
                cov_X = np.cov(np.hstack((vectori_mahala, vectorj_mahala)))
                dis = distance.mahalanobis(vectori, vectorj, VI=cov_X)
                if dis <= 1e-5:
                    weight = dis_max
                else:
                    weight = 1.0 / dis
                if weight >= epsw:
                    edge = (i, j, {'weight': weight})
                    edges_list.append(edge)

    G = nx.Graph()
    G.add_edges_from(edges_list)
    return G, np.array(dis_all)
# G, dis_all = get_nxGraph_mahalanobis(X, epsw=1.0 / 220.0)
# simPlotGraph(G)
# get_distributions(dis_all)


# # rbf_kernel Gaussian Similarity
# X_nm, Y_nm = vectori[np.newaxis, :], vectorj[np.newaxis, :]# array of shape (n_samples_X, n_features)
# dis_rbf = rbf_kernel(X=X_nm, Y=Y_nm, gamma=0.025)[0, 0]# gamma need to be setup.
def get_nxGraph_rbf_kernel(X, epsw=1/0.01, gamma=0.025):
    (m, n) = X.shape
    dis_all = []
    edges_list = []
    for i in range(m):
        for j in range(m):
            if i != j:
                vectori, vectorj = X[i, :], X[j, :]
                # rbf_kernel Gaussian Similarity
                X_nm, Y_nm = vectori[np.newaxis, :], vectorj[np.newaxis, :]  # array of shape (n_samples_X, n_features)
                dis_rbf = rbf_kernel(X=X_nm, Y=Y_nm, gamma=gamma)[0, 0]  # gamma need to be setup.
                if dis_rbf < 1e-2:
                    dis_rbf = 1e-2
                dis_all.append(dis_rbf)
                weight = 1.0/dis_rbf
                if weight >= epsw:
                    edge = (i, j, {'weight': weight})
                    edges_list.append(edge)
    G = nx.Graph()
    G.add_edges_from(edges_list)
    return G, np.array(dis_all)
# G, dis_all = get_nxGraph_rbf_kernel(X, epsw=1/0.25, gamma = 0.025)
# simPlotGraph(G)
# get_distributions(dis_all)

# knn enn
def get_nxGraph_knn_enn(X, n_neighbors=5, radius=11, epsw=1.0 / 20.0):
    # knn enn
    # n_neighbors = 5
    # radius = 11
    # epsw = 0.5
    (m, n) = X.shape
    dis_all = []
    samples = X
    # algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'} , default='auto' Algorithm used to compute the nearest neighbors
    neigh = NearestNeighbors(n_neighbors=n_neighbors, radius=radius, algorithm='auto', leaf_size=30, metric='minkowski', p=2)# p=2 isequivalent to euclidean
    neigh.fit(samples)
    knn_edges_list = []
    enn_edges_list = []
    for i in range(m):
        v = X[i, :][np.newaxis, :]

        knn_dis, knn_node = neigh.kneighbors(X=v, n_neighbors=n_neighbors, return_distance=True)
        knn_dis, knn_node = list(knn_dis[0]), list(knn_node[0])
        dis_all += knn_dis
        index, ind = None, 0
        for nd in knn_node:
            if nd == i:
                index = ind
                break
            ind += 1
        if index is not None:
            knn_node.pop(index)
            knn_dis.pop(index)
        index = 0
        for neigh_v in knn_node:
            dis = knn_dis[index]
            if dis < 1e-5:
                weight = np.max(knn_dis)
            else:
                weight = 1.0/dis
            if weight >= epsw:
                edge = (i, neigh_v, {'weight': weight})
                knn_edges_list.append(edge)
            index += 1

        enn_dis, enn_node = neigh.radius_neighbors(X=v, radius=radius, return_distance=True)
        enn_dis, enn_node = list(enn_dis[0]), list(enn_node[0])
        index, ind = None, 0
        for nd in enn_node:
            if nd == i:
                index = ind
                break
            ind += 1
        if index is not None:
            enn_node.pop(index)
            enn_dis.pop(index)

        for neigh_v in enn_node:
            edge = (i, neigh_v, {'weight': 1.0})
            enn_edges_list.append(edge)

    knn_G = nx.Graph()
    knn_G.add_edges_from(knn_edges_list)
    enn_G = nx.Graph()
    enn_G.add_edges_from(enn_edges_list)

    return knn_G, enn_G, np.array(dis_all)
# knn_G, enn_G, dis_all = get_nxGraph_knn_enn(X, n_neighbors=5, radius=11, epsw=1.0/11.0)
# simPlotGraph(knn_G)
# simPlotGraph(enn_G)
# get_distributions(dis_all)
