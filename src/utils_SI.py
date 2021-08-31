import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from fitter import Fitter
import pandas as pd
from scipy.sparse import coo_matrix
from joblib import Parallel, delayed, effective_n_jobs

def get_coo_matrix_from_G(G, weight=None):
    row = np.array([u for u, _ in G.edges(data=False)])
    col = np.array([v for _, v in G.edges(data=False)])
    if weight is not None:
        data = np.array([w['weight'] for _, __, w in G.edges(data=True)])
    else:
        data = np.ones(shape=row.shape, dtype=np.float)
    coo_mat = coo_matrix((data, (row, col)), dtype=np.float)
    row = coo_mat.row
    col = coo_mat.col
    data = coo_mat.data
    return row, col, data

def get_Vigidegi(row, col, data, ndq):
    deg_ndq = {}  # ndq degrees
    nodes = []
    weights = []
    for nd in ndq:
        index_row = np.where(row == nd)
        index_col = np.where(col == nd)
        u = list(row[index_col]) + list(col[index_row])
        index_data = np.array(list(index_row[0]) + list(index_col[0]), dtype=np.int)
        w = list(data[index_data])
        deg_ndq[nd] = np.sum(w)
        nodes += u
        weights += w
    for nd in ndq:
        for _ in range(nodes.count(nd)):
            nodes.remove(nd)
    # V_G = np.sum(data) * 2.0
    deg_i = deg_ndq
    gi = len(nodes)
    Vi = np.sum(weights)
    return Vi, gi, deg_i

def get_Vgdeg(row, col, data, ndq_a, ndq_b):
    ndq = ndq_a + ndq_b
    L_X, L_Y, L_XY = len(ndq_a), len(ndq_b), len(ndq)
    deg_ndq = {}  # ndq degrees
    nodes_a, weights_a = [], []
    nodes_b, weights_b = [], []
    nodes, weights = [], []
    for nd in ndq[:L_X]:
        index_row = np.where(row == nd)
        index_col = np.where(col == nd)
        u = list(row[index_col]) + list(col[index_row])
        index_data = np.array(list(index_row[0]) + list(index_col[0]), dtype=np.int)
        w = list(data[index_data])
        deg_ndq[nd] = np.sum(w)
        nodes += u
        weights += w
    nodes_a += nodes
    weights_a += weights
    for nd in ndq[L_X:]:
        index_row = np.where(row == nd)
        index_col = np.where(col == nd)
        u = list(row[index_col]) + list(col[index_row])
        index_data = np.array(list(index_row[0]) + list(index_col[0]), dtype=np.int)
        w = list(data[index_data])
        deg_ndq[nd] = np.sum(w)
        nodes += u
        weights += w
    nodes_b += nodes[len(nodes_a):len(nodes)]
    weights_b += weights[len(weights_a):len(weights)]
    for nd in ndq[:L_X]:
        for _ in range(nodes.count(nd)):
            nodes.remove(nd)
        for _ in range(nodes_a.count(nd)):
            nodes_a.remove(nd)
    for nd in ndq[L_X:]:
        for _ in range(nodes.count(nd)):
            nodes.remove(nd)
        for _ in range(nodes_b.count(nd)):
            nodes_b.remove(nd)
    # V_G = np.sum(data) * 2.0
    deg_ij = deg_ndq
    deg_i = {nd: deg_ndq[nd] for nd in ndq_a}
    deg_j = {nd: deg_ndq[nd] for nd in ndq_b}
    gi = len(nodes_a)
    Vi = np.sum(weights_a)
    gj = len(nodes_b)
    Vj = np.sum(weights_b)
    gij = len(nodes)
    Vij = np.sum(weights)
    return Vi, Vj, Vij, gi, gj, gij, deg_i, deg_j, deg_ij

def deltDI_ij(row, col, data, ndq_a, ndq_b):
    ndq = ndq_a + ndq_b
    L_X, L_Y, L_XY = len(ndq_a), len(ndq_b), len(ndq)
    deg_ndq = {}  # ndq degrees
    nodes_a, weights_a = [], []
    nodes_b, weights_b = [], []
    nodes, weights = [], []
    for nd in ndq[:L_X]:
        index_row = np.where(row == nd)
        index_col = np.where(col == nd)
        u = list(row[index_col]) + list(col[index_row])
        index_data = np.array(list(index_row[0]) + list(index_col[0]), dtype=np.int)
        w = list(data[index_data])
        deg_ndq[nd] = np.sum(w)
        nodes += u
        weights += w
    nodes_a += nodes
    weights_a += weights
    for nd in ndq[L_X:]:
        index_row = np.where(row == nd)
        index_col = np.where(col == nd)
        u = list(row[index_col]) + list(col[index_row])
        index_data = np.array(list(index_row[0]) + list(index_col[0]), dtype=np.int)
        w = list(data[index_data])
        deg_ndq[nd] = np.sum(w)
        nodes += u
        weights += w
    nodes_b += nodes[len(nodes_a):len(nodes)]
    weights_b += weights[len(weights_a):len(weights)]

    for nd in ndq[:L_X]:
        for _ in range(nodes.count(nd)):
            nodes.remove(nd)
        for _ in range(nodes_a.count(nd)):
            nodes_a.remove(nd)
    for nd in ndq[L_X:]:
        for _ in range(nodes.count(nd)):
            nodes.remove(nd)
        for _ in range(nodes_b.count(nd)):
            nodes_b.remove(nd)
    # deg_ij = deg_ndq
    # deg_i = {nd: deg_ndq[nd] for nd in ndq_a}
    # deg_j = {nd: deg_ndq[nd] for nd in ndq_b}
    V_G = np.sum(data) * 2.0
    g_i = len(nodes_a)
    V_i = np.sum(weights_a)
    g_j = len(nodes_b)
    V_j = np.sum(weights_b)
    g_ij = len(nodes)
    V_ij = np.sum(weights)

    if V_i < 1e-5:
        V_i = 1.0
    if V_j < 1e-5:
        V_j = 1.0
    if V_ij < 1e-5:
        V_ij = 1.0
    if V_G < 1e-5:
        V_G = 1.0

    delt = -(V_i - g_i) * np.log2(V_i) - (V_j - g_j) * np.log2(V_j) \
           + (V_ij - g_ij) * np.log2(V_ij) \
           + (V_i + V_j - V_ij - g_i - g_j + g_ij) * np.log2(V_G)

    return delt / V_G

def parts_DI(row, col, data, ndq):
    deg_ndq = {}  # ndq degrees
    nodes = []
    weights = []
    for nd in ndq:
        index_row = np.where(row == nd)
        index_col = np.where(col == nd)
        u = list(row[index_col]) + list(col[index_row])
        index_data = np.array(list(index_row[0]) + list(index_col[0]), dtype=np.int)
        w = list(data[index_data])
        deg_ndq[nd] = np.sum(w)
        nodes += u
        weights += w
    for nd in ndq:
        for _ in range(nodes.count(nd)):
            nodes.remove(nd)
    V_G = np.sum(data) * 2.0
    # deg_i = deg_ndq
    gi = len(nodes)
    Vi = np.sum(weights)
    V_div = Vi / V_G
    V_div_hat = (Vi - gi) / V_G

    return 0.0 if V_div < 1e-5 else -V_div_hat * np.log2(V_div)

def producer(N):
    ndi = N[0]
    N.remove(ndi)
    for ndj in N:
        yield list(ndi), list(ndj)

def partition_producer(partition):
    for L in partition:
        yield L

def pLog2p(p_i, eps=1e-10):
    ind = np.where(p_i < eps)
    if len(ind) > 0:
        p_i[p_i < eps] = 1.0
        return p_i * np.log2(p_i)
    else:
        return p_i * np.log2(p_i)

def Log2p(p_i, eps=1e-10):
    ind = np.where(p_i < eps)
    if len(ind) > 0:
        p_i[p_i < eps] = 1.0
        return np.log2(p_i)
    else:
        return np.log2(p_i)

def get_oneStruInforEntropy(G, weight=None):
    G_du = G.degree(weight=weight)
    G_volume = sum(G_du[index_node] for index_node in G.nodes)
    G_pu_dic = {index_node: G_du[index_node] * 1.0 / (1.0 * G_volume) for index_node in G.nodes}
    G_pu = [G_pu_dic[index_node] for index_node in G.nodes]
    # Shonnon Entropy
    HP_G_Shonnon = sum(pLog2p(np.array(G_pu))) * (-1.0)
    return HP_G_Shonnon

# StruInforEntropy
def StruInforEntropy(G, partition, weight=None):
    # nodes = G.nodes
    # n = G.number_of_nodes()
    # m = G.number_of_edges()
    sub_set = partition.copy()
    degree = G.degree(weight=weight)
    G_volume = sum(degree[index_node] for index_node in G.nodes)
    Vij, gij, deg_ij = [], [], []
    for ind in range(len(sub_set)):
        sub_degree = 0
        dij = []
        for node in sub_set[ind]:
            sub_degree += degree[node]
            dij.append(degree[node])
        gj_c = nx.cut_size(G, sub_set[ind], weight=weight)
        Vij.append(sub_degree)
        gij.append(gj_c)
        deg_ij.append(np.array(dij))
    gij = np.array(gij, dtype=float)
    Vij = np.array(Vij, dtype=float)
    p_i = [deg_ij[i] / Vij[i] for i in range(len(Vij))]
    pLogp = [pLog2p(pi, eps=1e-10) for pi in p_i]
    sum_pLogp = np.array([np.sum(plp) for plp in pLogp], dtype=float)

    first = np.sum((-1.0) * Vij / (G_volume) * sum_pLogp)
    second = np.sum((-1.0) * gij / (G_volume) * Log2p(Vij / (G_volume)))

    HG = first + second

    return HG, Vij, gij, deg_ij

# StruInforEntropy
def get_oneStruInforEntropy_Partition(G, partition, weight=None):
    # nodes = G.nodes
    # n = G.number_of_nodes()
    # m = G.number_of_edges()
    sub_set = partition.copy()
    degree = G.degree(weight=weight)
    G_volume = sum(degree[index_node] for index_node in G.nodes)
    Vij, gij, deg_ij = [], [], []
    for ind in range(len(sub_set)):
        sub_degree = 0
        dij = []
        for node in sub_set[ind]:
            sub_degree += degree[node]
            dij.append(degree[node])
        gj_c = nx.cut_size(G, sub_set[ind], weight=weight)
        Vij.append(sub_degree)
        gij.append(gj_c)
        deg_ij.append(np.array(dij))
    gij = np.array(gij, dtype=float)
    Vij = np.array(Vij, dtype=float)
    p_i = [deg_ij[i] / Vij[i] for i in range(len(Vij))]
    pLogp = [pLog2p(pi, eps=1e-10) for pi in p_i]
    sum_pLogp = np.array([np.sum(plp) for plp in pLogp], dtype=float)

    first = np.sum((-1.0) * Vij / (G_volume) * sum_pLogp)
    second = np.sum((-1.0) * pLog2p(Vij / G_volume))

    oneHG = first + second

    return oneHG, Vij, gij, deg_ij

# # test: 2D-StruInforEntropy
# partition = [{0, 1, 2, 3}, {4, 5, 6, 7, 8}, {9, 10, 11, 12, 13}]
# HP_G, Vj, g_j, Gj_deg = StruInforEntropy(G, partition)

def delt_Xij(Xi, Xj, G, weight=None):
    Xij = Xi + Xj
    sub_set = [Xi, Xj, list(set(Xij))]
    degree = G.degree(weight=weight)
    G_volume = sum(degree[index_node] for index_node in G.nodes)
    Vij, gij = [], []
    for ind in range(len(sub_set)):
        sub_degree = 0
        for node in sub_set[ind]:
            sub_degree += degree[node]
        gj_c = nx.cut_size(G, sub_set[ind], weight=weight)
        Vij.append(sub_degree)
        gij.append(gj_c)
    gij = np.array(gij)
    Vij = np.array(Vij)
    g_i, g_j, g_ij = gij[0], gij[1], gij[2]
    V_i, V_j, V_ij = Vij[0], Vij[1], Vij[2]
    log_Vij = Log2p(Vij, eps=1e-10)
    delt_G_Pij = 1.0 / (G_volume) * ((V_i - g_i) * log_Vij[0] +
                                     (V_j - g_j) * log_Vij[1] -
                                     (V_ij - g_ij) * log_Vij[2] +
                                     (g_i + g_j - g_ij) * np.log2(G_volume + 1e-10))
    return delt_G_Pij

def delt_RXij(Xi, Xj, G, weight=None):
    Xij = Xi + Xj
    sub_set = [Xi, Xj, list(set(Xij))]
    # sub_set = [[0, 1], [2, 3], [0, 1, 2, 3]]
    # n = G.number_of_nodes()
    # m = G.number_of_edges()
    degree = G.degree(weight=weight)
    G_volume = sum(degree[index_node] for index_node in G.nodes) + 1e-10
    Vij, gij = [], []
    for ind in range(len(sub_set)):
        sub_degree = 0
        for node in sub_set[ind]:
            sub_degree += degree[node]
        gj_c = nx.cut_size(G, sub_set[ind], weight=weight)
        Vij.append(sub_degree)
        gij.append(gj_c)
    g_i, g_j, g_ij = gij[0], gij[1], gij[2]
    V_i, V_j, V_ij = Vij[0], Vij[1], Vij[2]
    # log_Vij = Log2p(Vij, eps=1e-10)

    RP_i = -(V_i - g_i) / G_volume * np.log2(V_i / G_volume + 1e-10)
    RP_j = -(V_j - g_j) / G_volume * np.log2(V_j / G_volume + 1e-10)
    RP_ij = -(V_ij - g_ij) / G_volume * np.log2(V_ij / G_volume + 1e-10)

    delt_G_RPij = RP_i + RP_j - RP_ij

    return delt_G_RPij

def doWhile(G, NodeA, weight=None):
    count = 0
    L = len(NodeA)
    for Xi in NodeA:
        NodeB = NodeA.copy()
        NodeB.remove(Xi)
        for Xj in NodeB:
            delt_ij = delt_Xij(Xi, Xj, G, weight=weight)  # 函数
            if delt_ij > 0:
                return True
    #         if delt_ij < 0:
    #             count += 1
    # if count - L * (L - 1) != 0:
    #     return True
    return False

def doRWhile(G, NodeA, weight=None):
    count = 0
    L = len(NodeA)
    for Xi in NodeA:
        NodeB = NodeA.copy()
        NodeB.remove(Xi)
        for Xj in NodeB:
            delt_ij = delt_RXij(Xi, Xj, G, weight=weight)  # 函数
            if delt_ij <= 0:
                return True
    #         if delt_ij > 0:
    #             count += 1
    # if count - L * (L - 1) != 0:
    #     return True
    return False

def doWhile_js(row, col, data, NodeA):
    count = 0
    L = len(NodeA)
    for Xi in NodeA:
        NodeB = NodeA.copy()
        NodeB.remove(Xi)
        for Xj in NodeB:
            delt_ij = deltDI_ij(row, col, data, ndq_a=Xi, ndq_b=Xj)  # 函数
            if delt_ij <= 0:
                return True
    #         if delt_ij > 0:
    #             count += 1
    # if count - L * (L - 1) != 0:
    #     return True
    return False


######################################算法主体min2DStruInforEntropyPartition -- 开始###########################################
def min2DStruInforEntropyPartition(G, weight=None):
    # Input 算法主体 -- 开始
    print("Partition by min2DHG ..........")
    nodes = list(G.nodes())
    nodes.sort()  # 节点编号升序排列
    global NodeA
    NodeA = [[node] for node in nodes]
    print("Init-Input:", NodeA)  # Input Data
    doWhileFlg = True
    NodeA.reverse()  # 逆序
    while doWhileFlg:
        Xi = NodeA.pop()
        Nj = NodeA.copy()
        delt_max = 0
        Xj_m = None
        for Xj in Nj:
            delt_ij = delt_Xij(Xi, Xj, G, weight=weight)  # 函数
            if delt_ij > 0 and delt_ij > delt_max:
                Xj_m = Xj
                delt_max = delt_ij
        if Xj_m in Nj and Xj_m is not None:
            Nj.remove(Xj_m)
            Xij = Xi + Xj_m
            Nj.insert(0, Xij)
            # print('Xi:', Xi, '+ Xj:', Xj_m, '-->', Xij, ' delt_ij_HG:', delt_max)
        elif Xj_m is None:
            Nj.insert(0, Xi)  # 首位值插入
        NodeA = Nj
        doWhileFlg = doWhile(G, NodeA, weight=weight)  # 是否继续循环
        # print(NodeA)
    # print('Output:', NodeA)
    sub_set = NodeA.copy()  # Final Result
    # Output: NodeA 算法主体 -- 结束

    # sort
    results = []
    for sb_result in sub_set:
        sb_result.sort()
        results.append(sb_result)
    results.sort()
    print('Output:', results)
    return results
######################################算法主体min2DStruInforEntropyPartition -- 结束###########################################

######################################算法主体maxDIPartition -- 开始###########################################
def maxDIPartition(G, weight=None, pk_partion=None):
    # Input 算法主体 -- 开始
    print("Partition by maxDI ..........")
    nodes = list(G.nodes())
    nodes.sort()  # 节点编号升序排列
    global NodeA
    if pk_partion is None:
        nodes = list(G.nodes())
        nodes.sort()  # 节点编号升序排列
        NodeA = [[node] for node in nodes]
    else:
        NodeA = pk_partion
    print("Init-Input:", NodeA)  # Input Data
    doWhileFlg = True
    NodeA.reverse()  # 逆序
    while doWhileFlg:
        Xi = NodeA.pop()
        Nj = NodeA.copy()
        delt_min = 0
        Xj_m = None
        for Xj in Nj:
            delt_ij = delt_RXij(Xi, Xj, G, weight=weight)  # 函数
            if delt_ij < 0 and delt_ij < delt_min:
                Xj_m = Xj
                delt_min = delt_ij
        if Xj_m in Nj and Xj_m is not None:
            Nj.remove(Xj_m)
            Xij = Xi + Xj_m
            Nj.insert(0, Xij)
            # print('Xi:', Xi, '+ Xj:', Xj_m, '-->', Xij, ' delt_RXij:', delt_min)
        elif Xj_m is None:
            Nj.insert(0, Xi)  # 首位值插入
        NodeA = Nj
        doWhileFlg = doRWhile(G, NodeA, weight=weight)  # 是否继续循环
        # print(NodeA)
    # print('Output:', NodeA)
    sub_set = NodeA.copy()  # Final Result
    # Output: NodeA 算法主体 -- 结束

    # sort
    results = []
    for sb_result in sub_set:
        sb_result.sort()
        results.append(sb_result)
    results.sort()
    print('Output:', results)
    return results
######################################算法主体maxDIPartition -- 结束###########################################

######################################算法主体maxDI 优化 -- 开始###########################################
def maxDI(G, weight=None, pk_partion=None):
    # Input 算法主体 -- 开始
    print("Partition by maxDI ..........")
    row, col, data = get_coo_matrix_from_G(G, weight=weight)
    nodes = list(G.nodes())
    nodes.sort()  # 节点编号升序排列
    global NodeA
    if pk_partion is None:
        nodes = list(G.nodes())
        nodes.sort()  # 节点编号升序排列
        NodeA = [[node] for node in nodes]
    else:
        NodeA = pk_partion
    print("Init-Input:", NodeA)  # Input Data
    doWhileFlg = True
    NodeA.reverse()  # 逆序
    while doWhileFlg:
        Xi = NodeA.pop()
        Nj = NodeA.copy()
        delt_min = 0
        Xj_m = None
        for Xj in Nj:
            delt_ij = deltDI_ij(row, col, data, ndq_a=Xi, ndq_b=Xj)  # 函数
            if delt_ij < 0 and delt_ij < delt_min:
                Xj_m = Xj
                delt_min = delt_ij
        if Xj_m in Nj and Xj_m is not None:
            Nj.remove(Xj_m)
            Xij = Xi + Xj_m
            Nj.insert(0, Xij)
            # print('Xi:', Xi, '+ Xj:', Xj_m, '-->', Xij, ' delt_RXij:', delt_min)
        elif Xj_m is None:
            Nj.insert(0, Xi)  # 首位值插入
        NodeA = Nj
        doWhileFlg = doWhile_js(row, col, data, NodeA)  # 是否继续循环
        # print(NodeA)
    # print('Output:', NodeA)
    sub_set = NodeA.copy()  # Final Result
    # Output: NodeA 算法主体 -- 结束

    # sort
    results = []
    for sb_result in sub_set:
        sb_result.sort()
        results.append(sb_result)
    results.sort()
    print('Output:', results)
    return results
######################################算法主体maxDI 优化 -- 结束###########################################

######################################算法主体maxDI 迭代优化-Parallel -- 开始###########################################
def iter_maxDI(G, iter_max=100, pk_partion=None, weight=None, n_jobs=4, verbose=0):
    # iter_max = 100
    # n_jobs = 4
    row, col, data = get_coo_matrix_from_G(G, weight=weight)
    global N
    if pk_partion is None:
        nodes = list(G.nodes())
        nodes.sort()  # 节点编号升序排列
        N = [[node] for node in nodes]
    else:
        N = pk_partion
    out = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(parts_DI)(row, col, data, P) for P in partition_producer(N))
    # print('(iter:%d ---> DI:%.2f bits)' % (0, np.sum(out)))
    DI = np.zeros(iter_max + 1)
    DI[0] = float('%.3f' % np.sum(out))
    print('(iter:%d ---> DI:%.3f bits)' % (0, DI[0]))
    for iter in range(iter_max):
        ndi = N[0]
        out = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(deltDI_ij)(row, col, data, ndi, ndj) for ndi, ndj in producer(N))
        out_min = min(out)
        if out_min < 0:
            ndj = N[out.index(out_min)]
            N.remove(ndj)
            N.append(ndi + ndj)
        elif ndi not in N:
            N.append(ndi)
        out = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(parts_DI)(row, col, data, P) for P in partition_producer(N))
        DI[iter + 1] = float('%.3f' % np.sum(out))
        if (iter + 1) % 50 == 0:
            print('(iter:%d ---> DI:%.3f bits)' % (iter+1, DI[iter+1]))
        # if (iter + 1) >= 2000 and np.var(DI[-2000:-1], ddof=1) < 1e-10:  # 计算样本方差 （ 计算时除以 N - 1 ）
        #     DI = DI[:iter + 2]
        #     break

    # sort results
    results = []
    for sb_result in N:
        sb_result.sort()
        results.append(sb_result)
    results.sort()

    return results, DI
######################################算法主体maxDI 迭代优化-Parallel -- 开始###########################################

# get_DI by Parallel
def get_DI_Parallel(G, partion=None, weight=None, n_jobs=4, verbose=0):
    row, col, data = get_coo_matrix_from_G(G, weight=weight)
    out = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(parts_DI)(row, col, data, P) for P in partition_producer(partion))
    DI = float('%.3f' % np.sum(out))
    return DI

# get_DI
def get_DI(G, weight=None):
    print("G: number_of_nodes:{0}, number_of_edges:{1}".format(G.number_of_nodes(), G.number_of_edges()))
    results = maxDIPartition(G, weight=weight)
    twoHG, Vj, g_j, Gj_deg = StruInforEntropy(G, partition=results, weight=weight)
    oneHG = get_oneStruInforEntropy(G, weight=weight)
    DI = oneHG - twoHG
    return DI

# n, m = G.number_of_nodes(), G.number_of_edges()
# Eg_List = [(u, v, w['weight']) for (u, v, w) in G.edges(data=True)]
def create_weight_graph_txt_file(G, filename='InputGraph'):
    file = open(filename + '.txt', mode='w')
    str_number_of_nodes = '{0}\n'.format(G.number_of_nodes())
    file.write(str_number_of_nodes)
    for (u, v, w) in G.edges(data=True):
        str_edge = '{0} {1} {2}\n'.format(u, v, w['weight'])
        file.write(str_edge)
    file.close()

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

# label_result
def label_result(nsamples, result):
    y_result = np.zeros((nsamples,)) - 1  # -1 为噪声点
    cls = 0
    for p in result:
        y_result[p] = cls
        cls += 1
    return y_result

########################## Test #################################
def main():
    # 无权重图 or all weights == 1.0
    G = nx.Graph()
    edges_list = [(0, 1, {'weight': 1.0}),
                  (0, 2, {'weight': 1.0}),  # test: outer point
                  (0, 3, {'weight': 1.0}),
                  # (0, 4, {'weight': 1.0}),
                  (1, 2, {'weight': 1.0}),  # test: outer point
                  (1, 3, {'weight': 1.0}),
                  (2, 3, {'weight': 1.0}),  # test: outer point
                  (2, 4, {'weight': 1.0}),
                  (4, 5, {'weight': 1.0}),
                  (4, 7, {'weight': 1.0}),
                  (5, 6, {'weight': 1.0}),
                  (5, 7, {'weight': 1.0}),
                  (5, 8, {'weight': 1.0}),
                  (6, 8, {'weight': 1.0}),
                  (7, 8, {'weight': 1.0}),
                  (8, 9, {'weight': 1.0}),
                  (9, 10, {'weight': 1.0}),
                  (9, 12, {'weight': 1.0}),
                  (9, 13, {'weight': 1.0}),
                  (10, 11, {'weight': 1.0}),
                  (10, 12, {'weight': 1.0}),
                  (11, 12, {'weight': 1.0}),
                  (11, 13, {'weight': 1.0}),
                  (12, 13, {'weight': 1.0})]
    G.add_edges_from(edges_list)

    # plot graph
    plot_graph(G, with_labels=True)

    # a = [9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492,
    #      9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492,
    #      9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492,
    #      9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492,
    #      9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492,
    #      9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492,
    #      9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492,
    #      9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492,
    #      9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492,
    #      9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492,
    #      9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492,
    #      10.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492, 9.492]
    # total_var = np.var(a)  # 计算总体方差 （ 计算时除以样本数 N ）
    # sample_var = np.var(a, ddof=1)  # 计算样本方差 （ 计算时除以 N - 1 ）

    # # min2DStruInforEntropyPartition
    # results = min2DStruInforEntropyPartition(G, weight='weight')
    # print("Partition-Size(by min2DHG):", len(results))
    # print("Partition(by min2DHG):", results)
    # # 2D-StruInforEntropy
    # HG, Vj, g_j, Gj_deg = StruInforEntropy(G, partition=results, weight='weight')
    # oneHG = get_oneStruInforEntropy(G, weight='weight')
    # print("1DStruInforEntropy:", oneHG)
    # print("2DStruInforEntropy:", HG)
    # print('--'*15)

    # # maxDI
    # results = maxDIPartition(G, weight='weight')# 基础版
    # print("Partition-Size(by maxDI):", len(results))
    # print("Partition(by maxDI):", results)
    # # 2D-StruInforEntropy
    # HG, Vj, g_j, Gj_deg = StruInforEntropy(G, partition=results, weight='weight')
    # oneHG = get_oneStruInforEntropy(G, weight='weight')
    # print("1DStruInforEntropy:", oneHG)
    # print("2DStruInforEntropy:", HG)
    # print("maxDI:", oneHG-HG)
    # oneHG, Vj, g_j, Gj_deg = get_oneStruInforEntropy_Partition(G, partition=results, weight='weight')
    # print("1DStruInforEntropy_Partition:", oneHG)

    # print('--'*15)
    # results = [[0, 1, 2, 3], [4, 5, 6, 7, 8], [9, 10, 11, 12, 13]]
    # print(results)
    # HG, Vj, g_j, Gj_deg = StruInforEntropy(G, partition=results, weight='weight')
    # print("2DStruInforEntropy:", HG)
    # print('Vj', Vj)
    # print('g_j', g_j)
    # print('--'*15)

    edges_di = []
    edges = [(u, v, w['weight']) for u, v, w in G.edges(data=True)]
    row, col, data = get_coo_matrix_from_G(G, weight='weight')
    for edge in edges:
        # DI_ij = delt_RXij(Xi=[edge[0]], Xj=[edge[1]], G=G, weight='weight')
        DI_ij = deltDI_ij(row, col, data, ndq_a=[edge[0]], ndq_b=[edge[1]])
        DI_ij = float("%.3f" % (DI_ij))
        delt_di = ((edge[0], edge[1]), edge[2], DI_ij)
        edges_di.append(delt_di)
        print(delt_di)

    # print('--' * 15)
    # results, DI = iter_maxDI(G, iter_max=50, pk_partion=None, weight='weight', n_jobs=4, verbose=0)  # 并行加速
    # print(results)
    # print(DI[-1])
    # print(DI)
    # print('--' * 15)

    # results, DI = get_iter_maxDICluster(G, iter_max=50)  # 并行加速
    # print(results)
    # print(DI[-1])
    # print(DI)
    # print('--' * 15)

    # maxDI
    results = maxDI(G, weight='weight')  # 优化版
    print("Partition-Size(by maxDI):", len(results))
    # 2D-StruInforEntropy
    HG, Vj, g_j, Gj_deg = StruInforEntropy(G, partition=results, weight='weight')
    oneHG = get_oneStruInforEntropy(G, weight='weight')
    print("1DStruInforEntropy:", oneHG)
    print("2DStruInforEntropy:", HG)

    # row, col, data = get_coo_matrix_from_G(G, weight='weight')
    # ndq_a = [0, 1, 2, 3]
    # ndq_b = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    # Vi, Vj, Vij, gi, gj, gij, deg_i, deg_j, deg_ij = get_Vgdeg(row, col, data, ndq_a, ndq_b)
    # V_G = np.sum(data) * 2.0
    # ndq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    # Vi, gi, deg_i = get_Vigidegi(row, col, data, ndq)

if __name__ == '__main__':
    main()