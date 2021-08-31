import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from utils_SI import get_DI
from utils_graphs import plot_graph
from utils_graphs import simPlotGraph, get_distributions
from utils_graphs import get_nxGraph_minkowski, get_nxGraph_euclidean, get_nxGraph_mahalanobis, get_nxGraph_manhattan
from utils_graphs import get_nxGraph_chebyshev, get_nxGraph_canberra, get_nxGraph_cosine, get_nxGraph_pearsonr
from utils_graphs import get_nxGraph_rbf_kernel, get_nxGraph_knn_enn

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

def matrix2gray(mat, seg=[1]):
    """
    convert matrix image into gray image
    :param mat: matrix (n x m) with numpy type
    :return:  mat
    """
    min_v = int(mat.min())
    max_v = int(mat.max())

    # 0--255
    gray = (mat - min_v) / (max_v - min_v) * 255
    # 颜色分段
    node_seg = (np.array(seg) - min_v) / (max_v - min_v) * 255

    return gray.astype(np.uint8), node_seg.astype(np.uint8)
    
def gray2rgb(gray, node_seg, color_dict):
    """
    convert c into RGB image
    :param gray: single channel image with numpy type
    :param color_dict: color map
    :return:  rgb image
    """
    point1 = node_seg[0]
    # point2 = node_seg[0]
    # point3 = node_seg[0]
    # point4 = node_seg[0]
    # 1：创建新图像容器
    rgb_image = np.zeros(shape=(*gray.shape, 3))
    # 2： 遍历每个像素点
    for i in range(rgb_image.shape[0]):
        for j in range(rgb_image.shape[1]):
            # 3：对不同的灰度值选择不同的颜色
            if gray[i, j] < point1:
                rgb_image[i, j, :] = color_dict["color1"]
            else:
                rgb_image[i, j, :] = color_dict["color2"]
    return rgb_image.astype(np.uint8)
    
def save_adjacency_matrix_png(G, save=False, filename='./rgb.png'):
    A = np.array(nx.adjacency_matrix(G).todense())
    # 颜色
    color_dict = {"color1": [255, 255, 255],
                  "color2": [103, 103, 103]}
    # 0--255
    gray, node_seg = matrix2gray(mat=A, seg=[1])
    # gray2rgb保存成png图像
    rgb_image = gray2rgb(gray, node_seg, color_dict)
    # 保存成彩色图片
    plt.imshow(rgb_image)  # Needs to be in row, col order
    filename = filename# = './rgb.png'
    # plt.axis('off')
    ticks = [j for j in range(G.number_of_nodes())]
    labels = [j + 1 for j in range(G.number_of_nodes())]
    plt.yticks(ticks=ticks, labels=labels)
    plt.xticks(ticks=ticks, labels=labels)
    plt.tick_params(labelsize=10)
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    if save:
        plt.savefig(filename, dpi=600, transparent=True)
    plt.show()
    return A

# dyna_name='SherringtonKirkpatrickIsing', 'BranchingModel', 'IsingGlauber'
def get_Simulate_Dynamics_TS(G, L = 2001, dyna_name='SherringtonKirkpatrickIsing', savefig=False, name_str=''):
    import netrd
    # dictionary of some example dynamics to play around with
    dynamics = {'BranchingModel': netrd.dynamics.BranchingModel(),
                'IsingGlauber': netrd.dynamics.IsingGlauber(),
                'SherringtonKirkpatrickIsing': netrd.dynamics.SherringtonKirkpatrickIsing()
                }
    # select the dynamical process you want to simulate on the network
    dyna_name = dyna_name
    DY = dynamics[dyna_name]

    # how long should the time series be?
    # L = 2001
    # simulate DY dynamics on the network
    TS = DY.simulate(G, L)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=300)
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    ax.imshow(TS, cmap='Greys', aspect='auto')
    ax.set_yticks([])
    ax.set_xlabel("Time(s)", size=14)
    ax.set_ylabel("Node ID", size=14)
    ax.set_title("%s dynamics" % dyna_name, size=14)

    if savefig:
        plt.savefig('./data/' + dyna_name + '_dynamic_'+name_str+'.svg', bbox_inches='tight', dpi=600, transparent=True)
    plt.show()

    return TS

def get_Reconstruction_from_Simulate(G, L = 2001, dyna_name='SherringtonKirkpatrickIsing', savefig=False, name_str=''):
    import netrd
    import itertools as it
    # dictionary of some example dynamics to play around with
    dynamics = {'BranchingModel':              netrd.dynamics.BranchingModel(),
                'IsingGlauber':                netrd.dynamics.IsingGlauber(),
                'SherringtonKirkpatrickIsing': netrd.dynamics.SherringtonKirkpatrickIsing()
                }

    # select the dynamical process you want to simulate on the network
    # dyna_name = 'SherringtonKirkpatrickIsing'
    dyna_name = dyna_name
    DY = dynamics[dyna_name]

    # how long should the time series be?
    # L = 2001
    L = L
    # simulate DY dynamics on the network
    TS = DY.simulate(G, L)

    fig, ax = plt.subplots(1, 3, figsize=(18, 4), gridspec_kw={'width_ratios': [1, 1, 2.15]}, dpi=200)
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    pos = nx.kamada_kawai_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='w', linewidths=2.5, edgecolors='.2', ax=ax[0])
    nx.draw_networkx_edges(G, pos, width=1.75, edge_color='.6', alpha=0.5, ax=ax[0])

    ax[0].set_title("Ground truth graph")
    ax[0].set_axis_off()

    # ax[1].imshow(nx.to_numpy_array(G), cmap='Greys', aspect='auto')
    # ax[1].set_yticks([])
    # ax[1].set_xticks([])
    # ax[1].set_title("Ground truth adjaency matrix")
    # or:
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='w', linewidths=2.5, edgecolors='.2', ax=ax[1])
    nx.draw_networkx_edges(G, pos, width=0.0, edge_color=None, alpha=0.0, ax=ax[1])
    ax[1].set_title("Neglect of graph links")
    ax[1].set_axis_off()

    ax[2].imshow(TS, cmap='Greys', aspect='auto')
    ax[2].set_yticks([])
    ax[2].set_xlabel("Time(s)", size=14)
    ax[2].set_ylabel("Node ID", size=14)
    # ax[2].set_title("%s dynamics" % dyna_name, size=14)
    ax[2].set_title("Simulate KIM dynamics", size=14)

    if savefig:
        plt.savefig('./data/'+name_str+'_Ground-truth-network.svg', bbox_inches='tight', dpi=600, transparent=True)
    plt.show()

    # get average degree of the network
    k_avg = np.mean(list(dict(G.degree()).values()))
    # dictionary of some of the reconstruction techniques
    recons = {
        'CorrelationMatrix':            netrd.reconstruction.CorrelationMatrix(),
        # 'FreeEnergyMinimization':       netrd.reconstruction.FreeEnergyMinimization(),# ok
        'GraphicalLasso':               netrd.reconstruction.GraphicalLasso(),
        # 'ThoulessAndersonPalmer':       netrd.reconstruction.ThoulessAndersonPalmer(),# ok
        'MutualInformationMatrix': netrd.reconstruction.MutualInformationMatrix(),
        'MaximumLikelihoodEstimation':  netrd.reconstruction.MaximumLikelihoodEstimation()# ok
        }
    # for ease of visualization, we'll threshold all the reconstructions
    kwargs = {'threshold_type': 'degree', 'avg_k': k_avg}
    # dictionary to store the outputs
    Wdict = {}
    GraphDict = {}
    # loop over all the reconstruction techniques
    for ri, R1 in list(recons.items()):
        # print(ri)
        R1.fit(TS, **kwargs)
        Wr = R1.results['thresholded_matrix']
        Wdict[ri] = Wr
        G = R1.results['graph']
        if isinstance(G, type(nx.DiGraph())):
            G = G.to_undirected(reciprocal=True)
        GraphDict[ri] = G

    w = 6.0
    h = 6.0
    ncols = 2
    nrows = 2
    tups = list(it.product(range(nrows), range(ncols)))

    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * w, nrows * h), dpi=200)
    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    plt.rcParams['font.sans-serif'] = 'Times New Roman'

    ix = 0
    for lab_i, W_i in Wdict.items():
        a = ax[tups[ix]]
        a.imshow(W_i, cmap='Greys')
        ix += 1
        a.set_title(lab_i, color='.0', fontsize='medium')
        a.set_yticks([])
        a.set_xticks([])
    if savefig:
        plt.savefig('./data/'+name_str+'_reconstruction.svg', bbox_inches='tight', dpi=600, transparent=True)
    plt.show()
    return GraphDict, Wdict

def get_dis_Graphs(G1, GraphDict, Wdict, weight=None, savefig=False, name_str=''):
    import netrd
    if savefig:
        di_g_0 = get_DI(G1, weight=weight)
        str_di = 'DI(' + name_str + '_GroundTruth):' + str(di_g_0)
        di_file = r'./data/' + name_str + '_DI.txt'
        di_txt_file = open(di_file, mode='w')
        di_txt_file.write(str_di)
        di_txt_file.write('\n')

    # dictionary of some of the graph distance measures in netrd
    # (leaving out some of the more computationally expensive ones)
    him_dis = netrd.distance.HammingIpsenMikhailov()
    Plotdict = {}
    Plotdict['CorrelationMatrix'] = Wdict['CorrelationMatrix']# 0, 0
    Plotdict['GraphicalLasso'] = Wdict['GraphicalLasso']# 0, 1
    Plotdict['GroundTruthMatrix'] = nx.adjacency_matrix(G1).todense()# 0, 2
    Plotdict['MutualInformationMatrix'] = Wdict['MutualInformationMatrix']# 1, 0
    Plotdict['MaximumLikelihoodEstimation'] = Wdict['MaximumLikelihoodEstimation']# 1, 1
    DISdict = {}# 1, 2
    if savefig:
        dis_file = r'./data/'+name_str+'_dis.txt'
        txt_file = open(dis_file, mode='w')
    for g_lab, g_i in GraphDict.items():
        # print(g_lab)
        G2 = GraphDict[g_lab]
        distance = him_dis.dist(G1, G2)
        DISdict[g_lab] = distance

        if savefig:
            print(name_str + '_' + g_lab + ' DI............')
            G3 = nx.Graph()
            edges_list = [(u, v, {'weight': 1.0}) for (u, v, w) in G2.edges(data=True)]# this is a trick
            G3.add_edges_from(edges_list)
            di_g = get_DI(G3, weight=weight)

            str_di = 'DI(' + name_str + '_' + g_lab + '):' + str(di_g)
            di_txt_file.write(str_di)
            di_txt_file.write('\n')

            str_v = 'Dis('+name_str + ', ' + g_lab + '):' + str(distance)
            txt_file.write(str_v)
            txt_file.write('\n')

        # G1、G2 plot
        fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=200, gridspec_kw={'width_ratios': [1, 1]})
        plt.subplots_adjust(wspace=0.1)
        plt.rcParams['font.sans-serif'] = 'Times New Roman'
        pos1 = nx.kamada_kawai_layout(G1)
        pos2 = nx.kamada_kawai_layout(G2)
        # G1 plot
        nx.draw_networkx_nodes(G1, pos1, node_size=40, linewidths=1.5, edgecolors='.3', node_color='lightskyblue', ax=ax[0])
        nx.draw_networkx_edges(G1, pos1, width=2, alpha=0.6, edge_color='.6', ax=ax[0])
        ax[0].set_title(r'Ground Truth Graph G', fontsize='x-large', color='.3')
        ax[0].set_axis_off()
        # G2 plot
        nx.draw_networkx_nodes(G2, pos2, node_size=40, linewidths=1.5, edgecolors='.3', node_color='lightcoral', ax=ax[1])
        nx.draw_networkx_edges(G2, pos2, width=2, alpha=0.6, edge_color='.6', ax=ax[1])
        ax[1].set_title(r'Extracted Graph G (by %s)'% g_lab, fontsize='x-large', color='.3')
        ax[1].set_axis_off()
        if savefig:
            plt.savefig('./data/'+name_str + '_' + g_lab + '_outG1G2.png', bbox_inches='tight', dpi=600, transparent=True)
        plt.show()
    if savefig:
        txt_file.close()
        di_txt_file.close()

    for lab_i, W_i in Plotdict.items():
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), dpi=200)
        plt.rcParams['font.sans-serif'] = 'Times New Roman'
        ax.imshow(W_i, cmap='Greys')
        ax.set_title(lab_i, color='.0', fontsize='medium')
        ax.set_yticks([])
        ax.set_xticks([])
        if savefig:
            plt.savefig('./data/'+name_str + '_' + lab_i + '_out.png', bbox_inches='tight', dpi=600, transparent=True)
        plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 3), dpi=300)
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    ax.bar(x=list(range(len(DISdict))), height=list(DISdict.values()), color='#A0A0A0', fc='w', ec='.2', lw=1.5)
    ax.set_xticks(list(range(len(DISdict))))
    ax.set_xticklabels(["%02i" % (i + 1) for i in list(range(len(DISdict)))], fontsize='small')
    ax.set_xlim(-0.6, len(GraphDict) - 0.4)
    ax.set_yscale('log')
    ax.set_ylabel(r'HimDis(G, G*)', fontsize='x-large', color='.2')
    ax.tick_params(labelbottom=True, bottom=True, labelleft=True, left=True, labelcolor='.4', color='.3')
    ax.grid(linewidth=1.25, color='.75', alpha=0.25)
    for i, title_i in enumerate(list(GraphDict.keys())):
        lab_i = "%02i - %s" % (i + 1, title_i)
        ax.text(1.02, 1 - (i / len(GraphDict)) - 0.55 / len(GraphDict), lab_i, ha='left', va='center',
                   color='.3', transform=ax.transAxes, fontsize='small')
    if savefig:
        plt.savefig('./data/'+name_str + '_6_out.png', bbox_inches='tight', dpi=600, transparent=True)
    plt.show()


def get_himdis_Graphs(G1, GraphDict, weight=None, savefig=False, name_dis='', name_str=''):
    import netrd
    if savefig:
        di_g_0 = get_DI(G1, weight=weight)
        di_g_0_str = "%.3f" % (di_g_0)
        str_di = 'DI(' + name_dis + '_' + name_str + '_GroundTruth):' + di_g_0_str
        di_file = r'./data/'+ name_dis + '_' + name_str + '_DI.txt'
        di_txt_file = open(di_file, mode='w')
        di_txt_file.write(str_di)
        di_txt_file.write('\n')

    # dictionary of some of the graph distance measures in netrd
    # (leaving out some of the more computationally expensive ones)
    him_dis = netrd.distance.HammingIpsenMikhailov()
    DISdict = {}# 1, 2
    if savefig:
        dis_file = r'./data/' + name_dis + '_' +name_str+'_dis.txt'
        txt_file = open(dis_file, mode='w')
    for g_lab, g_i in GraphDict.items():
        print(g_lab)
        G2 = GraphDict[g_lab]
        distance = him_dis.dist(G1, G2)
        DISdict[g_lab] = distance

        if savefig:
            print(name_str + '_' + g_lab + ' DI............')
            G3 = nx.Graph()
            edges_list = [(u, v, {'weight': 1.0}) for (u, v, w) in G2.edges(data=True)]# this is a trick
            G3.add_edges_from(edges_list)
            di_g = get_DI(G3, weight=weight)

            di_g_str = "%.3f" % (di_g)
            str_di = 'DI('+ name_dis + '_' + name_str + '_' + g_lab + '):' + di_g_str
            di_txt_file.write(str_di)
            di_txt_file.write('\n')

            distance_str = "%.3f" % (distance)
            str_v = 'Dis('+ name_dis + '_' +name_str + ', ' + g_lab + '):' + distance_str
            txt_file.write(str_v)
            txt_file.write('\n')

        # G1、G2 plot
        fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=200, gridspec_kw={'width_ratios': [1, 1]})
        plt.subplots_adjust(wspace=0.1)
        plt.rcParams['font.sans-serif'] = 'Times New Roman'
        pos1 = nx.kamada_kawai_layout(G1)
        pos2 = nx.kamada_kawai_layout(G2)
        # G1 plot
        nx.draw_networkx_nodes(G1, pos1, node_size=40, linewidths=1.5, edgecolors='.3', node_color='lightskyblue', ax=ax[0])
        nx.draw_networkx_edges(G1, pos1, width=2, alpha=0.6, edge_color='.6', ax=ax[0])
        ax[0].set_title(r'Ground Truth Graph G', fontsize='x-large', color='.3')
        ax[0].set_axis_off()
        # G2 plot
        nx.draw_networkx_nodes(G2, pos2, node_size=40, linewidths=1.5, edgecolors='.3', node_color='lightcoral', ax=ax[1])
        nx.draw_networkx_edges(G2, pos2, width=2, alpha=0.6, edge_color='.6', ax=ax[1])
        ax[1].set_title(r'Extracted Graph G (by %s)'% g_lab, fontsize='x-large', color='.3')
        ax[1].set_axis_off()
        if savefig:
            plt.savefig('./data/'+name_str + '_' + g_lab + '_outG1G2.png', bbox_inches='tight', dpi=600, transparent=True)
        plt.show()
    if savefig:
        txt_file.close()
        di_txt_file.close()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 3), dpi=300)
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    ax.bar(x=list(range(len(DISdict))), height=list(DISdict.values()), color='#A0A0A0', fc='w', ec='.2', lw=1.5)
    ax.set_xticks(list(range(len(DISdict))))
    ax.set_xticklabels(["%02i" % (i + 1) for i in list(range(len(DISdict)))], fontsize='small')
    ax.set_xlim(-0.6, len(GraphDict) - 0.4)
    ax.set_yscale('log')
    ax.set_ylabel(r'HimDis(G, G*)', fontsize='x-large', color='.2')
    ax.tick_params(labelbottom=True, bottom=True, labelleft=True, left=True, labelcolor='.4', color='.3')
    ax.grid(linewidth=1.25, color='.75', alpha=0.25)
    for i, title_i in enumerate(list(GraphDict.keys())):
        lab_i = "%02i - %s" % (i + 1, title_i)
        ax.text(1.02, 1 - (i / len(GraphDict)) - 0.55 / len(GraphDict), lab_i, ha='left', va='center',
                   color='.3', transform=ax.transAxes, fontsize='small')
    if savefig:
        plt.savefig('./data/'+name_str + '_6_out.png', bbox_inches='tight', dpi=600, transparent=True)
    plt.show()


def get_DI_2Graphs(G1, G2, weight=None, title_name='DI'):
    di_g_0 = get_DI(G1, weight=weight)
    di_g = get_DI(G2, weight=weight)
    DIdict = {}
    DIdict['Ground Truth Graph DI'] = di_g_0
    DIdict['Extracted Graph DI'] = di_g

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 3), dpi=300)
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    ax.bar(x=list(range(len(DIdict))), height=list(DIdict.values()), color='#A0A0A0', fc='w', ec='.2', lw=1.5)
    ax.set_xticks(list(range(len(DIdict))))
    ax.set_xticklabels(["%02i" % (i + 1) for i in list(range(len(DIdict)))], fontsize='small')
    ax.set_xlim(-0.6, len(DIdict) - 0.4)
    ax.set_title(title_name)
    ax.set_yscale('log')
    ax.set_ylabel(r'DI', fontsize='x-large', color='.2')
    ax.tick_params(labelbottom=True, bottom=True, labelleft=True, left=True, labelcolor='.4', color='.3')
    ax.grid(linewidth=1.25, color='.75', alpha=0.25)
    for i, title_i in enumerate(list(DIdict.keys())):
        lab_i = "%02i - %s" % (i + 1, title_i)
        ax.text(1.02, 1 - (i / len(DIdict)) - 0.55 / len(DIdict), lab_i, ha='left', va='center',
                   color='.3', transform=ax.transAxes, fontsize='small')
    plt.show()

def main():
    #################################### First Experiment ####################################
    save_flg = False
    # a. ring of cliques - 团图 OK
    Ga = nx.generators.ring_of_cliques(num_cliques=6, clique_size=5)
    plot_graph(Ga, save=save_flg, filename='./data/ring_clique.svg')
    A = save_adjacency_matrix_png(Ga, save=save_flg, filename='./data/ring_clique_rgb.png')
    # X = generate_synthetic_data(A)
    GraphDict, Wdict = get_Reconstruction_from_Simulate(Ga, L=2001, dyna_name='SherringtonKirkpatrickIsing', savefig=save_flg, name_str='ring_clique')
    # GraphDict['MutualInformationMatrix'] ---> 'CorrelationMatrix', 'FreeEnergyMinimization', 'GraphicalLasso', 'ThoulessAndersonPalmer', 'MaximumLikelihoodEstimation', 'MutualInformationMatrix'
    get_dis_Graphs(G1=Ga, GraphDict=GraphDict, Wdict=Wdict, savefig=save_flg, name_str='ring_clique')
    # b. *n*-dimensional grid graph OK
    Gb = nx.grid_graph(dim=[1, 6, 5])
    plot_graph(Gb, save=save_flg, filename='./data/grid.svg')
    A = save_adjacency_matrix_png(Gb, save=save_flg, filename='./data/grid_rgb.png')
    # X = generate_synthetic_data(A)
    GraphDict, Wdict = get_Reconstruction_from_Simulate(Gb, L=2001, dyna_name='SherringtonKirkpatrickIsing', savefig=save_flg, name_str='grid')
    # GraphDict['MutualInformationMatrix'] ---> 'CorrelationMatrix', 'FreeEnergyMinimization', 'GraphicalLasso', 'ThoulessAndersonPalmer', 'MaximumLikelihoodEstimation', 'MutualInformationMatrix'
    get_dis_Graphs(G1=Gb, GraphDict=GraphDict, Wdict=Wdict, savefig=save_flg, name_str='grid')
    # c. Scale-free graph(B-A) OK
    Gc = nx.barabasi_albert_graph(30, 1)# generate BA network
    plot_graph(Gc, save=save_flg, filename='./data/barabasi_albert.svg')
    A = save_adjacency_matrix_png(Gc, save=save_flg, filename='./data/barabasi_albert_rgb.png')
    # X = generate_synthetic_data(A)
    GraphDict, Wdict = get_Reconstruction_from_Simulate(Gc, L=2001, dyna_name='SherringtonKirkpatrickIsing', savefig=save_flg, name_str='barabasi_albert')
    # GraphDict['MutualInformationMatrix'] ---> 'CorrelationMatrix', 'FreeEnergyMinimization', 'GraphicalLasso', 'ThoulessAndersonPalmer', 'MaximumLikelihoodEstimation', 'MutualInformationMatrix'
    get_dis_Graphs(G1=Gc, GraphDict=GraphDict, Wdict=Wdict, savefig=save_flg, name_str='barabasi_albert')
    #################################### First Experiment End ####################################

    #################################### Second Experiment ####################################
    # a. ring of cliques - 团图 OK
    Ga = nx.generators.ring_of_cliques(num_cliques=6, clique_size=5)
    plot_graph(Ga, save=save_flg, filename='./data/ring_clique.svg')
    # dyna_name='SherringtonKirkpatrickIsing', 'BranchingModel', 'IsingGlauber'
    TS_1a = get_Simulate_Dynamics_TS(Ga, L=2001, dyna_name='SherringtonKirkpatrickIsing', savefig=save_flg, name_str='ring_clique')
    TS_2a = get_Simulate_Dynamics_TS(Ga, L=2001, dyna_name='BranchingModel', savefig=save_flg, name_str='ring_clique')
    TS_3a = get_Simulate_Dynamics_TS(Ga, L=2001, dyna_name='IsingGlauber', savefig=save_flg, name_str='ring_clique')
    # b. *n*-dimensional grid graph OK
    Gb = nx.grid_graph(dim=[1, 6, 5])
    plot_graph(Gb, save=save_flg, filename='./data/grid.svg')
    # dyna_name='SherringtonKirkpatrickIsing', 'BranchingModel', 'IsingGlauber'
    TS_1b = get_Simulate_Dynamics_TS(Gb, L=2001, dyna_name='SherringtonKirkpatrickIsing', savefig=save_flg, name_str='grid')
    TS_2b = get_Simulate_Dynamics_TS(Gb, L=2001, dyna_name='BranchingModel', savefig=save_flg, name_str='grid')
    TS_3b = get_Simulate_Dynamics_TS(Gb, L=2001, dyna_name='IsingGlauber', savefig=save_flg, name_str='grid')
    # c. Scale-free graph(B-A) OK
    Gc = nx.barabasi_albert_graph(30, 1)# generate BA network
    plot_graph(Gc, save=save_flg, filename='./data/barabasi_albert.svg')
    # dyna_name='SherringtonKirkpatrickIsing', 'BranchingModel', 'IsingGlauber'
    TS_1c = get_Simulate_Dynamics_TS(Gc, L=2001, dyna_name='SherringtonKirkpatrickIsing', savefig=save_flg, name_str='barabasi_albert')
    TS_2c = get_Simulate_Dynamics_TS(Gc, L=2001, dyna_name='BranchingModel', savefig=save_flg, name_str='barabasi_albert')
    TS_3c = get_Simulate_Dynamics_TS(Gc, L=2001, dyna_name='IsingGlauber', savefig=save_flg, name_str='barabasi_albert')

    # ---------------------------------------euclidean--------------------------------------------
    # euclidean
    # a. graph
    GraphDict = {}
    G, dis_all = get_nxGraph_euclidean(X=TS_1a, epsw=1.0 / 62.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='euclidean', name_str='TS_1a')
    simPlotGraph(G)
    get_DI_2Graphs(Ga, G, weight='weight', title_name='TS_1a')
    GraphDict['euclidean_TS_1a'] = G
    G, dis_all = get_nxGraph_euclidean(X=TS_2a, epsw=1.0 / 13.5)
    get_distributions(dis_all, savefig=save_flg, name_dis='euclidean', name_str='TS_2a')
    simPlotGraph(G)
    get_DI_2Graphs(Ga, G, weight='weight', title_name='TS_2a')
    GraphDict['euclidean_TS_2a'] = G
    G, dis_all = get_nxGraph_euclidean(X=TS_3a, epsw=1.0 / 31.5)
    get_distributions(dis_all, savefig=save_flg, name_dis='euclidean', name_str='TS_3a')
    simPlotGraph(G)
    get_DI_2Graphs(Ga, G, weight='weight', title_name='TS_3a')
    GraphDict['euclidean_TS_3a'] = G
    get_himdis_Graphs(G1=Ga, GraphDict=GraphDict, savefig=save_flg, name_dis='euclidean', name_str='ring_clique')

    # b. graph
    GraphDict = {}
    G, dis_all = get_nxGraph_euclidean(X=TS_1b, epsw=1.0 / 63)
    get_distributions(dis_all, savefig=save_flg, name_dis='euclidean', name_str='TS_1b')
    simPlotGraph(G)
    get_DI_2Graphs(Gb, G, weight='weight', title_name='TS_1b')
    GraphDict['euclidean_TS_1b'] = G
    G, dis_all = get_nxGraph_euclidean(X=TS_2b, epsw=1.0 / 13.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='euclidean', name_str='TS_2b')
    simPlotGraph(G)
    get_DI_2Graphs(Gb, G, weight='weight', title_name='TS_2b')
    GraphDict['euclidean_TS_2b'] = G
    G, dis_all = get_nxGraph_euclidean(X=TS_3b, epsw=1.0 / 31.5)
    get_distributions(dis_all, savefig=save_flg, name_dis='euclidean', name_str='TS_3b')
    simPlotGraph(G)
    get_DI_2Graphs(Gb, G, weight='weight', title_name='TS_3b')
    GraphDict['euclidean_TS_3b'] = G
    get_himdis_Graphs(G1=Gb, GraphDict=GraphDict, savefig=save_flg, name_dis='euclidean', name_str='grid')

    # c. graph
    GraphDict = {}
    G, dis_all = get_nxGraph_euclidean(X=TS_1c, epsw=1.0 / 62.5)
    get_distributions(dis_all, savefig=save_flg, name_dis='euclidean', name_str='TS_1c')
    simPlotGraph(G)
    get_DI_2Graphs(Gc, G, weight='weight', title_name='TS_1c')
    GraphDict['euclidean_TS_1c'] = G
    G, dis_all = get_nxGraph_euclidean(X=TS_2c, epsw=1.0 / 23.5)
    get_distributions(dis_all, savefig=save_flg, name_dis='euclidean', name_str='TS_2c')
    simPlotGraph(G)
    get_DI_2Graphs(Gc, G, weight='weight', title_name='TS_2c')
    GraphDict['euclidean_TS_2c'] = G
    G, dis_all = get_nxGraph_euclidean(X=TS_3c, epsw=1.0 / 31.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='euclidean', name_str='TS_3c')
    simPlotGraph(G)
    get_DI_2Graphs(Gc, G, weight='weight', title_name='TS_3c')
    GraphDict['euclidean_TS_3c'] = G
    get_himdis_Graphs(G1=Gc, GraphDict=GraphDict, savefig=save_flg, name_dis='euclidean', name_str='barabasi_albert')
    # ---------------------------------------euclidean end--------------------------------------------

    # ---------------------------------------mahalanobis--------------------------------------------
    # mahalanobis
    # a. graph
    GraphDict = {}
    G, dis_all = get_nxGraph_mahalanobis(X=TS_1a, epsw=1.0 / 2900.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='mahalanobis', name_str='TS_1a')
    simPlotGraph(G)
    get_DI_2Graphs(Ga, G, weight='weight', title_name='TS_1a')
    GraphDict['mahalanobis_TS_1a'] = G
    G, dis_all = get_nxGraph_mahalanobis(X=TS_2a, epsw=1.0 / 130.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='mahalanobis', name_str='TS_2a')
    simPlotGraph(G)
    get_DI_2Graphs(Ga, G, weight='weight', title_name='TS_2a')
    GraphDict['mahalanobis_TS_2a'] = G
    G, dis_all = get_nxGraph_mahalanobis(X=TS_3a, epsw=1.0 / 720.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='mahalanobis', name_str='TS_3a')
    simPlotGraph(G)
    get_DI_2Graphs(Ga, G, weight='weight', title_name='TS_3a')
    GraphDict['mahalanobis_TS_3a'] = G
    get_himdis_Graphs(G1=Ga, GraphDict=GraphDict, savefig=save_flg, name_dis='mahalanobis', name_str='ring_clique')

    # b. graph
    GraphDict = {}
    G, dis_all = get_nxGraph_mahalanobis(X=TS_1b, epsw=1.0 / 2900.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='mahalanobis', name_str='TS_1b')
    simPlotGraph(G)
    get_DI_2Graphs(Gb, G, weight='weight', title_name='TS_1b')
    GraphDict['mahalanobis_TS_1b'] = G
    G, dis_all = get_nxGraph_mahalanobis(X=TS_2b, epsw=1.0 / 130.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='mahalanobis', name_str='TS_2b')
    simPlotGraph(G)
    get_DI_2Graphs(Gb, G, weight='weight', title_name='TS_2b')
    GraphDict['mahalanobis_TS_2b'] = G
    G, dis_all = get_nxGraph_mahalanobis(X=TS_3b, epsw=1.0 / 720.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='mahalanobis', name_str='TS_3b')
    simPlotGraph(G)
    get_DI_2Graphs(Gb, G, weight='weight', title_name='TS_3b')
    GraphDict['mahalanobis_TS_3b'] = G
    get_himdis_Graphs(G1=Gb, GraphDict=GraphDict, savefig=save_flg, name_dis='mahalanobis', name_str='grid')

    # c. graph
    GraphDict = {}
    G, dis_all = get_nxGraph_mahalanobis(X=TS_1c, epsw=1.0 / 2800.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='mahalanobis', name_str='TS_1c')
    simPlotGraph(G)
    get_DI_2Graphs(Gc, G, weight='weight', title_name='TS_1c')
    GraphDict['mahalanobis_TS_1c'] = G
    G, dis_all = get_nxGraph_mahalanobis(X=TS_2c, epsw=1.0 / 300.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='mahalanobis', name_str='TS_2c')
    simPlotGraph(G)
    get_DI_2Graphs(Gc, G, weight='weight', title_name='TS_2c')
    GraphDict['mahalanobis_TS_2c'] = G
    G, dis_all = get_nxGraph_mahalanobis(X=TS_3c, epsw=1.0 / 700.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='mahalanobis', name_str='TS_3c')
    simPlotGraph(G)
    get_DI_2Graphs(Gc, G, weight='weight', title_name='TS_3c')
    GraphDict['mahalanobis_TS_3c'] = G
    get_himdis_Graphs(G1=Gc, GraphDict=GraphDict, savefig=save_flg, name_dis='mahalanobis', name_str='barabasi_albert')
    # ---------------------------------------euclidean end--------------------------------------------

    # ---------------------------------------minkowski--------------------------------------------
    # minkowski
    # a. graph
    GraphDict = {}
    G, dis_all = get_nxGraph_minkowski(X=TS_1a, epsw=1.0 / 20.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='minkowski', name_str='TS_1a')
    simPlotGraph(G)
    get_DI_2Graphs(Ga, G, weight='weight', title_name='TS_1a')
    GraphDict['minkowski_TS_1a'] = G
    G, dis_all = get_nxGraph_minkowski(X=TS_2a, epsw=1.0 / 5.6)
    get_distributions(dis_all, savefig=save_flg, name_dis='minkowski', name_str='TS_2a')
    simPlotGraph(G)
    get_DI_2Graphs(Ga, G, weight='weight', title_name='TS_2a')
    GraphDict['minkowski_TS_2a'] = G
    G, dis_all = get_nxGraph_minkowski(X=TS_3a, epsw=1.0 / 9.9)
    get_distributions(dis_all, savefig=save_flg, name_dis='minkowski', name_str='TS_3a')
    simPlotGraph(G)
    get_DI_2Graphs(Ga, G, weight='weight', title_name='TS_3a')
    GraphDict['minkowski_TS_3a'] = G
    get_himdis_Graphs(G1=Ga, GraphDict=GraphDict, savefig=save_flg, name_dis='minkowski', name_str='ring_clique')

    # b. graph
    GraphDict = {}
    G, dis_all = get_nxGraph_minkowski(X=TS_1b, epsw=1.0 / 19.65)
    get_distributions(dis_all, savefig=save_flg, name_dis='minkowski', name_str='TS_1b')
    simPlotGraph(G)
    get_DI_2Graphs(Gb, G, weight='weight', title_name='TS_1b')
    GraphDict['minkowski_TS_1b'] = G
    G, dis_all = get_nxGraph_minkowski(X=TS_2b, epsw=1.0 / 5.6)
    get_distributions(dis_all, savefig=save_flg, name_dis='minkowski', name_str='TS_2b')
    simPlotGraph(G)
    get_DI_2Graphs(Gb, G, weight='weight', title_name='TS_2b')
    GraphDict['minkowski_TS_2b'] = G
    G, dis_all = get_nxGraph_minkowski(X=TS_3b, epsw=1.0 / 9.95)
    get_distributions(dis_all, savefig=save_flg, name_dis='minkowski', name_str='TS_3b')
    simPlotGraph(G)
    get_DI_2Graphs(Gb, G, weight='weight', title_name='TS_3b')
    GraphDict['minkowski_TS_3b'] = G
    get_himdis_Graphs(G1=Gb, GraphDict=GraphDict, savefig=save_flg, name_dis='minkowski', name_str='grid')

    # c. graph
    GraphDict = {}
    G, dis_all = get_nxGraph_minkowski(X=TS_1c, epsw=1.0 / 19.5)
    get_distributions(dis_all, savefig=save_flg, name_dis='minkowski', name_str='TS_1c')
    simPlotGraph(G)
    get_DI_2Graphs(Gc, G, weight='weight', title_name='TS_1c')
    GraphDict['minkowski_TS_1c'] = G
    G, dis_all = get_nxGraph_minkowski(X=TS_2c, epsw=1.0 / 7.5)
    get_distributions(dis_all, savefig=save_flg, name_dis='minkowski', name_str='TS_2c')
    simPlotGraph(G)
    get_DI_2Graphs(Gc, G, weight='weight', title_name='TS_2c')
    GraphDict['minkowski_TS_2c'] = G
    G, dis_all = get_nxGraph_minkowski(X=TS_3c, epsw=1.0 / 10.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='minkowski', name_str='TS_3c')
    simPlotGraph(G)
    get_DI_2Graphs(Gc, G, weight='weight', title_name='TS_3c')
    GraphDict['minkowski_TS_3c'] = G
    get_himdis_Graphs(G1=Gc, GraphDict=GraphDict, savefig=save_flg, name_dis='minkowski', name_str='barabasi_albert')
    # ---------------------------------------minkowski end--------------------------------------------

    # ---------------------------------------manhattan--------------------------------------------
    # manhattan
    # a. graph
    GraphDict = {}
    G, dis_all = get_nxGraph_manhattan(X=TS_1a, epsw=1.0 / 2050.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='manhattan', name_str='TS_1a')
    simPlotGraph(G)
    get_DI_2Graphs(Ga, G, weight='weight', title_name='TS_1a')
    GraphDict['manhattan_TS_1a'] = G
    G, dis_all = get_nxGraph_manhattan(X=TS_2a, epsw=1.0 / 185.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='manhattan', name_str='TS_2a')
    simPlotGraph(G)
    get_DI_2Graphs(Ga, G, weight='weight', title_name='TS_2a')
    GraphDict['manhattan_TS_2a'] = G
    G, dis_all = get_nxGraph_manhattan(X=TS_3a, epsw=1.0 / 1025.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='manhattan', name_str='TS_3a')
    simPlotGraph(G)
    get_DI_2Graphs(Ga, G, weight='weight', title_name='TS_3a')
    GraphDict['manhattan_TS_3a'] = G
    get_himdis_Graphs(G1=Ga, GraphDict=GraphDict, savefig=save_flg, name_dis='manhattan', name_str='ring_clique')

    # b. graph
    GraphDict = {}
    G, dis_all = get_nxGraph_manhattan(X=TS_1b, epsw=1.0 / 2050.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='manhattan', name_str='TS_1b')
    simPlotGraph(G)
    get_DI_2Graphs(Gb, G, weight='weight', title_name='TS_1b')
    GraphDict['manhattan_TS_1b'] = G
    G, dis_all = get_nxGraph_manhattan(X=TS_2b, epsw=1.0 / 175.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='manhattan', name_str='TS_2b')
    simPlotGraph(G)
    get_DI_2Graphs(Gb, G, weight='weight', title_name='TS_2b')
    GraphDict['manhattan_TS_2b'] = G
    G, dis_all = get_nxGraph_manhattan(X=TS_3b, epsw=1.0 / 1025.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='manhattan', name_str='TS_3b')
    simPlotGraph(G)
    get_DI_2Graphs(Gb, G, weight='weight', title_name='TS_3b')
    GraphDict['manhattan_TS_3b'] = G
    get_himdis_Graphs(G1=Gb, GraphDict=GraphDict, savefig=save_flg, name_dis='manhattan', name_str='grid')

    # c. graph
    GraphDict = {}
    G, dis_all = get_nxGraph_manhattan(X=TS_1c, epsw=1.0 / 2000.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='manhattan', name_str='TS_1c')
    simPlotGraph(G)
    get_DI_2Graphs(Gc, G, weight='weight', title_name='TS_1c')
    GraphDict['manhattan_TS_1c'] = G
    G, dis_all = get_nxGraph_manhattan(X=TS_2c, epsw=1.0 / 450.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='manhattan', name_str='TS_2c')
    simPlotGraph(G)
    get_DI_2Graphs(Gc, G, weight='weight', title_name='TS_2c')
    GraphDict['manhattan_TS_2c'] = G
    G, dis_all = get_nxGraph_manhattan(X=TS_3c, epsw=1.0 / 980.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='manhattan', name_str='TS_3c')
    simPlotGraph(G)
    get_DI_2Graphs(Gc, G, weight='weight', title_name='TS_3c')
    GraphDict['manhattan_TS_3c'] = G
    get_himdis_Graphs(G1=Gc, GraphDict=GraphDict, savefig=save_flg, name_dis='manhattan', name_str='barabasi_albert')
    # ---------------------------------------manhattan end--------------------------------------------


    # ---------------------------------------chebyshev--------------------------------------------
    # chebyshev
    # a. graph
    GraphDict = {}
    G, dis_all = get_nxGraph_chebyshev(X=TS_1a, epsw=1.0 / 2.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='chebyshev', name_str='TS_1a')
    simPlotGraph(G)
    get_DI_2Graphs(Ga, G, weight='weight', title_name='TS_1a')
    GraphDict['chebyshev_TS_1a'] = G
    G, dis_all = get_nxGraph_chebyshev(X=TS_2a, epsw=1.0 / 11.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='chebyshev', name_str='TS_2a')
    simPlotGraph(G)
    get_DI_2Graphs(Ga, G, weight='weight', title_name='TS_2a')
    GraphDict['chebyshev_TS_2a'] = G
    G, dis_all = get_nxGraph_chebyshev(X=TS_3a, epsw=1.0 / 1.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='chebyshev', name_str='TS_3a')
    simPlotGraph(G)
    get_DI_2Graphs(Ga, G, weight='weight', title_name='TS_3a')
    GraphDict['chebyshev_TS_3a'] = G
    get_himdis_Graphs(G1=Ga, GraphDict=GraphDict, savefig=save_flg, name_dis='chebyshev', name_str='ring_clique')

    # b. graph
    GraphDict = {}
    G, dis_all = get_nxGraph_chebyshev(X=TS_1b, epsw=1.0 / 2.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='chebyshev', name_str='TS_1b')
    simPlotGraph(G)
    get_DI_2Graphs(Gb, G, weight='weight', title_name='TS_1b')
    GraphDict['chebyshev_TS_1b'] = G
    G, dis_all = get_nxGraph_chebyshev(X=TS_2b, epsw=1.0 / 11.5)
    get_distributions(dis_all, savefig=save_flg, name_dis='chebyshev', name_str='TS_2b')
    simPlotGraph(G)
    get_DI_2Graphs(Gb, G, weight='weight', title_name='TS_2b')
    GraphDict['chebyshev_TS_2b'] = G
    G, dis_all = get_nxGraph_chebyshev(X=TS_3b, epsw=1.0 / 1.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='chebyshev', name_str='TS_3b')
    simPlotGraph(G)
    get_DI_2Graphs(Gb, G, weight='weight', title_name='TS_3b')
    GraphDict['chebyshev_TS_3b'] = G
    get_himdis_Graphs(G1=Gb, GraphDict=GraphDict, savefig=save_flg, name_dis='chebyshev', name_str='grid')

    # c. graph
    GraphDict = {}
    G, dis_all = get_nxGraph_chebyshev(X=TS_1c, epsw=1.0 / 2.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='chebyshev', name_str='TS_1c')
    simPlotGraph(G)
    get_DI_2Graphs(Gc, G, weight='weight', title_name='TS_1c')
    GraphDict['chebyshev_TS_1c'] = G
    G, dis_all = get_nxGraph_chebyshev(X=TS_2c, epsw=1.0 / 11.5)
    get_distributions(dis_all, savefig=save_flg, name_dis='chebyshev', name_str='TS_2c')
    simPlotGraph(G)
    get_DI_2Graphs(Gc, G, weight='weight', title_name='TS_2c')
    GraphDict['chebyshev_TS_2c'] = G
    G, dis_all = get_nxGraph_chebyshev(X=TS_3c, epsw=1.0 / 1.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='chebyshev', name_str='TS_3c')
    simPlotGraph(G)
    get_DI_2Graphs(Gc, G, weight='weight', title_name='TS_3c')
    GraphDict['chebyshev_TS_3c'] = G
    get_himdis_Graphs(G1=Gc, GraphDict=GraphDict, savefig=save_flg, name_dis='chebyshev', name_str='barabasi_albert')
    # ---------------------------------------chebyshev end--------------------------------------------

    # ---------------------------------------canberra--------------------------------------------
    # canberra
    # a. graph
    GraphDict = {}
    G, dis_all = get_nxGraph_canberra(X=TS_1a, epsw=1.0 / 1000.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='canberra', name_str='TS_1a')
    simPlotGraph(G)
    get_DI_2Graphs(Ga, G, weight='weight', title_name='TS_1a')
    GraphDict['canberra_TS_1a'] = G
    G, dis_all = get_nxGraph_canberra(X=TS_2a, epsw=1.0 / 1450.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='canberra', name_str='TS_2a')
    simPlotGraph(G)
    get_DI_2Graphs(Ga, G, weight='weight', title_name='TS_2a')
    GraphDict['canberra_TS_2a'] = G
    G, dis_all = get_nxGraph_canberra(X=TS_3a, epsw=1.0 / 1025.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='canberra', name_str='TS_3a')
    simPlotGraph(G)
    get_DI_2Graphs(Ga, G, weight='weight', title_name='TS_3a')
    GraphDict['canberra_TS_3a'] = G
    get_himdis_Graphs(G1=Ga, GraphDict=GraphDict, savefig=save_flg, name_dis='canberra', name_str='ring_clique')

    # b. graph
    GraphDict = {}
    G, dis_all = get_nxGraph_canberra(X=TS_1b, epsw=1.0 / 1025.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='canberra', name_str='TS_1b')
    simPlotGraph(G)
    get_DI_2Graphs(Gb, G, weight='weight', title_name='TS_1b')
    GraphDict['canberra_TS_1b'] = G
    G, dis_all = get_nxGraph_canberra(X=TS_2b, epsw=1.0 / 1450.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='canberra', name_str='TS_2b')
    simPlotGraph(G)
    get_DI_2Graphs(Gb, G, weight='weight', title_name='TS_2b')
    GraphDict['canberra_TS_2b'] = G
    G, dis_all = get_nxGraph_canberra(X=TS_3b, epsw=1.0 / 1025.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='canberra', name_str='TS_3b')
    simPlotGraph(G)
    get_DI_2Graphs(Gb, G, weight='weight', title_name='TS_3b')
    GraphDict['canberra_TS_3b'] = G
    get_himdis_Graphs(G1=Gb, GraphDict=GraphDict, savefig=save_flg, name_dis='canberra', name_str='grid')

    # c. graph
    GraphDict = {}
    G, dis_all = get_nxGraph_canberra(X=TS_1c, epsw=1.0 / 1000.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='canberra', name_str='TS_1c')
    simPlotGraph(G)
    get_DI_2Graphs(Gc, G, weight='weight', title_name='TS_1c')
    GraphDict['canberra_TS_1c'] = G
    G, dis_all = get_nxGraph_canberra(X=TS_2c, epsw=1.0 / 1450.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='canberra', name_str='TS_2c')
    simPlotGraph(G)
    get_DI_2Graphs(Gc, G, weight='weight', title_name='TS_2c')
    GraphDict['canberra_TS_2c'] = G
    G, dis_all = get_nxGraph_canberra(X=TS_3c, epsw=1.0 / 950.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='canberra', name_str='TS_3c')
    simPlotGraph(G)
    get_DI_2Graphs(Gc, G, weight='weight', title_name='TS_3c')
    GraphDict['canberra_TS_3c'] = G
    get_himdis_Graphs(G1=Gc, GraphDict=GraphDict, savefig=save_flg, name_dis='canberra', name_str='barabasi_albert')
    # ---------------------------------------canberra end--------------------------------------------

    # ---------------------------------------cosine--------------------------------------------
    # cosine
    # a. graph
    GraphDict = {}
    G, dis_all = get_nxGraph_cosine(X=TS_1a, epsw=0.5)
    get_distributions(dis_all, savefig=save_flg, name_dis='cosine', name_str='TS_1a')
    simPlotGraph(G)
    get_DI_2Graphs(Ga, G, weight='weight', title_name='TS_1a')
    GraphDict['cosine_TS_1a'] = G
    G, dis_all = get_nxGraph_cosine(X=TS_2a, epsw=0.5)
    get_distributions(dis_all, savefig=save_flg, name_dis='cosine', name_str='TS_2a')
    simPlotGraph(G)
    get_DI_2Graphs(Ga, G, weight='weight', title_name='TS_2a')
    GraphDict['cosine_TS_2a'] = G
    G, dis_all = get_nxGraph_cosine(X=TS_3a, epsw=0.45)
    get_distributions(dis_all, savefig=save_flg, name_dis='cosine', name_str='TS_3a')
    simPlotGraph(G)
    get_DI_2Graphs(Ga, G, weight='weight', title_name='TS_3a')
    GraphDict['cosine_TS_3a'] = G
    get_himdis_Graphs(G1=Ga, GraphDict=GraphDict, savefig=save_flg, name_dis='cosine', name_str='ring_clique')

    # b. graph
    GraphDict = {}
    G, dis_all = get_nxGraph_cosine(X=TS_1b, epsw=0.75)
    get_distributions(dis_all, savefig=save_flg, name_dis='cosine', name_str='TS_1b')
    simPlotGraph(G)
    get_DI_2Graphs(Gb, G, weight='weight', title_name='TS_1b')
    GraphDict['cosine_TS_1b'] = G
    G, dis_all = get_nxGraph_cosine(X=TS_2b, epsw=0.85)
    get_distributions(dis_all, savefig=save_flg, name_dis='cosine', name_str='TS_2b')
    simPlotGraph(G)
    get_DI_2Graphs(Gb, G, weight='weight', title_name='TS_2b')
    GraphDict['cosine_TS_2b'] = G
    G, dis_all = get_nxGraph_cosine(X=TS_3b, epsw=0.475)
    get_distributions(dis_all, savefig=save_flg, name_dis='cosine', name_str='TS_3b')
    simPlotGraph(G)
    get_DI_2Graphs(Gb, G, weight='weight', title_name='TS_3b')
    GraphDict['cosine_TS_3b'] = G
    get_himdis_Graphs(G1=Gb, GraphDict=GraphDict, savefig=save_flg, name_dis='cosine', name_str='grid')

    # c. graph
    GraphDict = {}
    G, dis_all = get_nxGraph_cosine(X=TS_1c, epsw=0.65)
    get_distributions(dis_all, savefig=save_flg, name_dis='cosine', name_str='TS_1c')
    simPlotGraph(G)
    get_DI_2Graphs(Gc, G, weight='weight', title_name='TS_1c')
    GraphDict['cosine_TS_1c'] = G
    G, dis_all = get_nxGraph_cosine(X=TS_2c, epsw=0.65)
    get_distributions(dis_all, savefig=save_flg, name_dis='cosine', name_str='TS_2c')
    simPlotGraph(G)
    get_DI_2Graphs(Gc, G, weight='weight', title_name='TS_2c')
    GraphDict['cosine_TS_2c'] = G
    G, dis_all = get_nxGraph_cosine(X=TS_3c, epsw=0.45)
    get_distributions(dis_all, savefig=save_flg, name_dis='cosine', name_str='TS_3c')
    simPlotGraph(G)
    get_DI_2Graphs(Gc, G, weight='weight', title_name='TS_3c')
    GraphDict['cosine_TS_3c'] = G
    get_himdis_Graphs(G1=Gc, GraphDict=GraphDict, savefig=save_flg, name_dis='cosine', name_str='barabasi_albert')
    # ---------------------------------------cosine end--------------------------------------------

    # ---------------------------------------pearsonr--------------------------------------------
    # pearsonr
    # a. graph
    GraphDict = {}
    G, dis_all = get_nxGraph_pearsonr(X=TS_1a, epsw=0.5)
    get_distributions(dis_all, savefig=save_flg, name_dis='pearsonr', name_str='TS_1a')
    simPlotGraph(G)
    get_DI_2Graphs(Ga, G, weight='weight', title_name='TS_1a')
    GraphDict['pearsonr_TS_1a'] = G
    G, dis_all = get_nxGraph_pearsonr(X=TS_2a, epsw=0.5)
    get_distributions(dis_all, savefig=save_flg, name_dis='pearsonr', name_str='TS_2a')
    simPlotGraph(G)
    get_DI_2Graphs(Ga, G, weight='weight', title_name='TS_2a')
    GraphDict['pearsonr_TS_2a'] = G
    G, dis_all = get_nxGraph_pearsonr(X=TS_3a, epsw=0.5)
    get_distributions(dis_all, savefig=save_flg, name_dis='pearsonr', name_str='TS_3a')
    simPlotGraph(G)
    get_DI_2Graphs(Ga, G, weight='weight', title_name='TS_3a')
    GraphDict['pearsonr_TS_3a'] = G
    get_himdis_Graphs(G1=Ga, GraphDict=GraphDict, savefig=save_flg, name_dis='pearsonr', name_str='ring_clique')

    # b. graph
    GraphDict = {}
    G, dis_all = get_nxGraph_pearsonr(X=TS_1b, epsw=0.5)
    get_distributions(dis_all, savefig=save_flg, name_dis='pearsonr', name_str='TS_1b')
    simPlotGraph(G)
    get_DI_2Graphs(Gb, G, weight='weight', title_name='TS_1b')
    GraphDict['pearsonr_TS_1b'] = G
    G, dis_all = get_nxGraph_pearsonr(X=TS_2b, epsw=0.5)
    get_distributions(dis_all, savefig=save_flg, name_dis='pearsonr', name_str='TS_2b')
    simPlotGraph(G)
    get_DI_2Graphs(Gb, G, weight='weight', title_name='TS_2b')
    GraphDict['pearsonr_TS_2b'] = G
    G, dis_all = get_nxGraph_pearsonr(X=TS_3b, epsw=0.5)
    get_distributions(dis_all, savefig=save_flg, name_dis='pearsonr', name_str='TS_3b')
    simPlotGraph(G)
    get_DI_2Graphs(Gb, G, weight='weight', title_name='TS_3b')
    GraphDict['pearsonr_TS_3b'] = G
    get_himdis_Graphs(G1=Gb, GraphDict=GraphDict, savefig=save_flg, name_dis='pearsonr', name_str='grid')

    # c. graph
    GraphDict = {}
    G, dis_all = get_nxGraph_pearsonr(X=TS_1c, epsw=0.5)
    get_distributions(dis_all, savefig=save_flg, name_dis='pearsonr', name_str='TS_1c')
    simPlotGraph(G)
    get_DI_2Graphs(Gc, G, weight='weight', title_name='TS_1c')
    GraphDict['pearsonr_TS_1c'] = G
    G, dis_all = get_nxGraph_pearsonr(X=TS_2c, epsw=0.15)
    get_distributions(dis_all, savefig=save_flg, name_dis='pearsonr', name_str='TS_2c')
    simPlotGraph(G)
    get_DI_2Graphs(Gc, G, weight='weight', title_name='TS_2c')
    GraphDict['pearsonr_TS_2c'] = G
    G, dis_all = get_nxGraph_pearsonr(X=TS_3c, epsw=0.5)
    get_distributions(dis_all, savefig=save_flg, name_dis='pearsonr', name_str='TS_3c')
    simPlotGraph(G)
    get_DI_2Graphs(Gc, G, weight='weight', title_name='TS_3c')
    GraphDict['pearsonr_TS_3c'] = G
    get_himdis_Graphs(G1=Gc, GraphDict=GraphDict, savefig=save_flg, name_dis='pearsonr', name_str='barabasi_albert')
    # ---------------------------------------pearsonr end--------------------------------------------

    # ---------------------------------------kernel--------------------------------------------
    # kernel
    # a. graph
    GraphDict = {}
    G, dis_all = get_nxGraph_rbf_kernel(X=TS_1a, epsw=0.0125, gamma=0.00125)
    get_distributions(dis_all, savefig=save_flg, name_dis='kernel', name_str='TS_1a')
    simPlotGraph(G)
    get_DI_2Graphs(Ga, G, weight='weight', title_name='TS_1a')
    GraphDict['kernel_TS_1a'] = G
    G, dis_all = get_nxGraph_rbf_kernel(X=TS_2a, epsw=0.6, gamma=0.0025)
    get_distributions(dis_all, savefig=save_flg, name_dis='kernel', name_str='TS_2a')
    simPlotGraph(G)
    get_DI_2Graphs(Ga, G, weight='weight', title_name='TS_2a')
    GraphDict['kernel_TS_2a'] = G
    G, dis_all = get_nxGraph_rbf_kernel(X=TS_3a, epsw=0.1, gamma=0.0025)
    get_distributions(dis_all, savefig=save_flg, name_dis='kernel', name_str='TS_3a')
    simPlotGraph(G)
    get_DI_2Graphs(Ga, G, weight='weight', title_name='TS_3a')
    GraphDict['kernel_TS_3a'] = G
    get_himdis_Graphs(G1=Ga, GraphDict=GraphDict, savefig=save_flg, name_dis='kernel', name_str='ring_clique')

    # b. graph
    GraphDict = {}
    G, dis_all = get_nxGraph_rbf_kernel(X=TS_1b, epsw=0.0125, gamma=0.00125)
    get_distributions(dis_all, savefig=save_flg, name_dis='kernel', name_str='TS_1b')
    simPlotGraph(G)
    get_DI_2Graphs(Gb, G, weight='weight', title_name='TS_1b')
    GraphDict['kernel_TS_1b'] = G
    G, dis_all = get_nxGraph_rbf_kernel(X=TS_2b, epsw=0.79, gamma=0.00125)
    get_distributions(dis_all, savefig=save_flg, name_dis='kernel', name_str='TS_2b')
    simPlotGraph(G)
    get_DI_2Graphs(Gb, G, weight='weight', title_name='TS_2b')
    GraphDict['kernel_TS_2b'] = G
    G, dis_all = get_nxGraph_rbf_kernel(X=TS_3b, epsw=0.1, gamma=0.0025)
    get_distributions(dis_all, savefig=save_flg, name_dis='kernel', name_str='TS_3b')
    simPlotGraph(G)
    get_DI_2Graphs(Gb, G, weight='weight', title_name='TS_3b')
    GraphDict['kernel_TS_3b'] = G
    get_himdis_Graphs(G1=Gb, GraphDict=GraphDict, savefig=save_flg, name_dis='kernel', name_str='grid')

    # c. graph
    GraphDict = {}
    G, dis_all = get_nxGraph_rbf_kernel(X=TS_1c, epsw=0.012, gamma=0.0025)
    get_distributions(dis_all, savefig=save_flg, name_dis='kernel', name_str='TS_1c')
    simPlotGraph(G)
    get_DI_2Graphs(Gc, G, weight='weight', title_name='TS_1c')
    GraphDict['kernel_TS_1c'] = G
    G, dis_all = get_nxGraph_rbf_kernel(X=TS_2c, epsw=0.45, gamma=0.0025)
    get_distributions(dis_all, savefig=save_flg, name_dis='kernel', name_str='TS_2c')
    simPlotGraph(G)
    get_DI_2Graphs(Gc, G, weight='weight', title_name='TS_2c')
    GraphDict['kernel_TS_2c'] = G
    G, dis_all = get_nxGraph_rbf_kernel(X=TS_3c, epsw=0.125, gamma=0.0025)
    get_distributions(dis_all, savefig=save_flg, name_dis='kernel', name_str='TS_3c')
    simPlotGraph(G)
    get_DI_2Graphs(Gc, G, weight='weight', title_name='TS_3c')
    GraphDict['kernel_TS_3c'] = G
    get_himdis_Graphs(G1=Gc, GraphDict=GraphDict, savefig=save_flg, name_dis='kernel', name_str='barabasi_albert')
    # ---------------------------------------kernel end--------------------------------------------


    # ---------------------------------------knn_enn--------------------------------------------
    # knn_enn
    # a. graph
    GraphDict = {}
    knn_G, enn_G, dis_all = get_nxGraph_knn_enn(X=TS_1a, n_neighbors=15, radius=11, epsw=1.0/60.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='knn_enn', name_str='TS_1a')
    simPlotGraph(knn_G)
    get_DI_2Graphs(Ga, knn_G, weight='weight', title_name='TS_1a')
    GraphDict['knn_TS_1a'] = knn_G
    knn_G, enn_G, dis_all = get_nxGraph_knn_enn(X=TS_2a, n_neighbors=15, radius=11, epsw=1.0/14.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='knn_enn', name_str='TS_2a')
    simPlotGraph(knn_G)
    get_DI_2Graphs(Ga, knn_G, weight='weight', title_name='TS_2a')
    GraphDict['knn_TS_2a'] = knn_G
    knn_G, enn_G, dis_all = get_nxGraph_knn_enn(X=TS_3a, n_neighbors=15, radius=11, epsw=1.0/31.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='knn_enn', name_str='TS_3a')
    simPlotGraph(knn_G)
    get_DI_2Graphs(Ga, knn_G, weight='weight', title_name='TS_3a')
    GraphDict['knn_TS_3a'] = knn_G
    get_himdis_Graphs(G1=Ga, GraphDict=GraphDict, savefig=save_flg, name_dis='knn_enn', name_str='ring_clique')

    # b. graph
    GraphDict = {}
    knn_G, enn_G, dis_all = get_nxGraph_knn_enn(X=TS_1b, n_neighbors=15, radius=11, epsw=1.0/62.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='knn_enn', name_str='TS_1b')
    simPlotGraph(knn_G)
    get_DI_2Graphs(Gb, knn_G, weight='weight', title_name='TS_1b')
    GraphDict['knn_TS_1b'] = knn_G
    knn_G, enn_G, dis_all = get_nxGraph_knn_enn(X=TS_2b, n_neighbors=15, radius=11, epsw=1.0/13.5)
    get_distributions(dis_all, savefig=save_flg, name_dis='knn_enn', name_str='TS_2b')
    simPlotGraph(knn_G)
    get_DI_2Graphs(Gb, knn_G, weight='weight', title_name='TS_2b')
    GraphDict['knn_TS_2b'] = knn_G
    knn_G, enn_G, dis_all = get_nxGraph_knn_enn(X=TS_3b, n_neighbors=15, radius=11, epsw=1.0/30.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='knn_enn', name_str='TS_3b')
    simPlotGraph(knn_G)
    get_DI_2Graphs(Gb, knn_G, weight='weight', title_name='TS_3b')
    GraphDict['knn_TS_3b'] = knn_G
    get_himdis_Graphs(G1=Gb, GraphDict=GraphDict, savefig=save_flg, name_dis='knn_enn', name_str='grid')

    # c. graph
    GraphDict = {}
    knn_G, enn_G, dis_all = get_nxGraph_knn_enn(X=TS_1c, n_neighbors=15, radius=11, epsw=1.0/62.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='knn_enn', name_str='TS_1c')
    simPlotGraph(knn_G)
    get_DI_2Graphs(Gc,knn_G, weight='weight', title_name='TS_1c')
    GraphDict['knn_TS_1c'] = knn_G
    knn_G, enn_G, dis_all = get_nxGraph_knn_enn(X=TS_2c, n_neighbors=15, radius=11, epsw=1.0/22.5)
    get_distributions(dis_all, savefig=save_flg, name_dis='knn_enn', name_str='TS_2c')
    simPlotGraph(knn_G)
    get_DI_2Graphs(Gc, knn_G, weight='weight', title_name='TS_2c')
    GraphDict['knn_TS_2c'] = knn_G
    knn_G, enn_G, dis_all = get_nxGraph_knn_enn(X=TS_3c, n_neighbors=15, radius=11, epsw=1.0/32.5)
    get_distributions(dis_all, savefig=save_flg, name_dis='knn_enn', name_str='TS_3c')
    simPlotGraph(knn_G)
    get_DI_2Graphs(Gc, knn_G, weight='weight', title_name='TS_3c')
    GraphDict['knn_TS_3c'] = knn_G
    get_himdis_Graphs(G1=Gc, GraphDict=GraphDict, savefig=save_flg, name_dis='knn_enn', name_str='barabasi_albert')
    # ---------------------------------------knn_enn end--------------------------------------------

    # ---------------------------------------knn_enn--------------------------------------------
    # knn_enn
    # a. graph
    GraphDict = {}
    knn_G, enn_G, dis_all = get_nxGraph_knn_enn(X=TS_1a, n_neighbors=15, radius=60, epsw=1.0/60.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='enn_enn', name_str='TS_1a')
    simPlotGraph(enn_G)
    get_DI_2Graphs(Ga, enn_G, weight='weight', title_name='TS_1a')
    GraphDict['enn_TS_1a'] = enn_G
    knn_G, enn_G, dis_all = get_nxGraph_knn_enn(X=TS_2a, n_neighbors=15, radius=14, epsw=1.0/14.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='enn_enn', name_str='TS_2a')
    simPlotGraph(enn_G)
    get_DI_2Graphs(Ga, enn_G, weight='weight', title_name='TS_2a')
    GraphDict['enn_TS_2a'] = enn_G
    knn_G, enn_G, dis_all = get_nxGraph_knn_enn(X=TS_3a, n_neighbors=15, radius=31, epsw=1.0/31.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='enn_enn', name_str='TS_3a')
    simPlotGraph(enn_G)
    get_DI_2Graphs(Ga, enn_G, weight='weight', title_name='TS_3a')
    GraphDict['enn_TS_3a'] = enn_G
    get_himdis_Graphs(G1=Ga, GraphDict=GraphDict, savefig=save_flg, name_dis='enn_enn', name_str='ring_clique')

    # b. graph
    GraphDict = {}
    knn_G, enn_G, dis_all = get_nxGraph_knn_enn(X=TS_1b, n_neighbors=15, radius=62, epsw=1.0/62.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='enn_enn', name_str='TS_1b')
    simPlotGraph(enn_G)
    get_DI_2Graphs(Gb, enn_G, weight='weight', title_name='TS_1b')
    GraphDict['enn_TS_1b'] = enn_G
    knn_G, enn_G, dis_all = get_nxGraph_knn_enn(X=TS_2b, n_neighbors=15, radius=13.5, epsw=1.0/13.5)
    get_distributions(dis_all, savefig=save_flg, name_dis='enn_enn', name_str='TS_2b')
    simPlotGraph(enn_G)
    get_DI_2Graphs(Gb, enn_G, weight='weight', title_name='TS_2b')
    GraphDict['enn_TS_2b'] = knn_G
    knn_G, enn_G, dis_all = get_nxGraph_knn_enn(X=TS_3b, n_neighbors=15, radius=30.0, epsw=1.0/30.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='enn_enn', name_str='TS_3b')
    simPlotGraph(enn_G)
    get_DI_2Graphs(Gb, enn_G, weight='weight', title_name='TS_3b')
    GraphDict['enn_TS_3b'] = enn_G
    get_himdis_Graphs(G1=Gb, GraphDict=GraphDict, savefig=save_flg, name_dis='enn_enn', name_str='grid')

    # c. graph
    GraphDict = {}
    knn_G, enn_G, dis_all = get_nxGraph_knn_enn(X=TS_1c, n_neighbors=15, radius=62.0, epsw=1.0/62.0)
    get_distributions(dis_all, savefig=save_flg, name_dis='enn_enn', name_str='TS_1c')
    simPlotGraph(knn_G)
    get_DI_2Graphs(Gc,knn_G, weight='weight', title_name='TS_1c')
    GraphDict['enn_TS_1c'] = knn_G
    knn_G, enn_G, dis_all = get_nxGraph_knn_enn(X=TS_2c, n_neighbors=15, radius=22.5, epsw=1.0/22.5)
    get_distributions(dis_all, savefig=save_flg, name_dis='enn_enn', name_str='TS_2c')
    simPlotGraph(knn_G)
    get_DI_2Graphs(Gc, knn_G, weight='weight', title_name='TS_2c')
    GraphDict['enn_TS_2c'] = knn_G
    knn_G, enn_G, dis_all = get_nxGraph_knn_enn(X=TS_3c, n_neighbors=15, radius=32.5, epsw=1.0/32.5)
    get_distributions(dis_all, savefig=save_flg, name_dis='enn_enn', name_str='TS_3c')
    simPlotGraph(knn_G)
    get_DI_2Graphs(Gc, knn_G, weight='weight', title_name='TS_3c')
    GraphDict['enn_TS_3c'] = knn_G
    get_himdis_Graphs(G1=Gc, GraphDict=GraphDict, savefig=save_flg, name_dis='enn_enn', name_str='barabasi_albert')
    # ---------------------------------------knn_enn end--------------------------------------------
    #################################### Second Experiment End ####################################

def third_EXP():
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from sklearn.cluster import SpectralClustering
    from MaxDICluster import MaxDIClustering

    save_flg = False
    #################################### Third Experiment ####################################
    # a. ring of cliques - 团图 OK
    Ga = nx.generators.ring_of_cliques(num_cliques=6, clique_size=5)
    plot_graph(Ga, save=save_flg, filename='./data/ring_clique.svg')
    # dyna_name='SherringtonKirkpatrickIsing', 'BranchingModel', 'IsingGlauber'
    TS_1a = get_Simulate_Dynamics_TS(Ga, L=2001, dyna_name='SherringtonKirkpatrickIsing', savefig=save_flg, name_str='ring_clique')
    TS_2a = get_Simulate_Dynamics_TS(Ga, L=2001, dyna_name='BranchingModel', savefig=save_flg, name_str='ring_clique')
    TS_3a = get_Simulate_Dynamics_TS(Ga, L=2001, dyna_name='IsingGlauber', savefig=save_flg, name_str='ring_clique')
    # b. *n*-dimensional grid graph OK
    Gb = nx.grid_graph(dim=[1, 6, 5])
    plot_graph(Gb, save=save_flg, filename='./data/grid.svg')
    # dyna_name='SherringtonKirkpatrickIsing', 'BranchingModel', 'IsingGlauber'
    TS_1b = get_Simulate_Dynamics_TS(Gb, L=2001, dyna_name='SherringtonKirkpatrickIsing', savefig=save_flg, name_str='grid')
    TS_2b = get_Simulate_Dynamics_TS(Gb, L=2001, dyna_name='BranchingModel', savefig=save_flg, name_str='grid')
    TS_3b = get_Simulate_Dynamics_TS(Gb, L=2001, dyna_name='IsingGlauber', savefig=save_flg, name_str='grid')
    # c. Scale-free graph(B-A) OK
    Gc = nx.barabasi_albert_graph(30, 1)# generate BA network
    plot_graph(Gc, save=save_flg, filename='./data/barabasi_albert.svg')
    # dyna_name='SherringtonKirkpatrickIsing', 'BranchingModel', 'IsingGlauber'
    TS_1c = get_Simulate_Dynamics_TS(Gc, L=2001, dyna_name='SherringtonKirkpatrickIsing', savefig=save_flg, name_str='barabasi_albert')
    TS_2c = get_Simulate_Dynamics_TS(Gc, L=2001, dyna_name='BranchingModel', savefig=save_flg, name_str='barabasi_albert')
    TS_3c = get_Simulate_Dynamics_TS(Gc, L=2001, dyna_name='IsingGlauber', savefig=save_flg, name_str='barabasi_albert')

    X = TS_1a

    # max_DI_cluster = MaxDIClustering(affinity='rbf', gamma=0.00125, threshold=0.008, show_graph=True)
    max_DI_cluster = MaxDIClustering(affinity='nearest_neighbors', n_neighbors=5, threshold=0.25, show_graph=True)
    max_DI_cluster.fit(X)
    y_pred = max_DI_cluster.labels_.astype(int)
    print(y_pred)
    silhouette_score = max_DI_cluster.silhouette_score
    calinski_harabasz_score = max_DI_cluster.calinski_harabasz_score
    print(silhouette_score)
    print(calinski_harabasz_score)

    algorithm = SpectralClustering(n_clusters=6, eigen_solver='arpack', n_neighbors=5, affinity="nearest_neighbors")
    algorithm.fit(X)
    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(int)
    else:
        y_pred = algorithm.predict(X)
    print(y_pred)


if __name__ == '__main__':
    main()
    # third_EXP()