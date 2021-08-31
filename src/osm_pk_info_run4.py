import time
import pickle
import numpy as np
import networkx as nx
from networkx import Graph
import pandas as pd
import geopandas as gpd
import xml.etree.cElementTree as et
from shapely.geometry import Point, MultiLineString
from osmnx import graph_from_xml, plot_graph, settings, config
from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN
from utils_SI import iter_maxDI, label_result, get_DI_Parallel, get_oneStruInforEntropy
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from pk_infor import get_PKResult

# # settings and config
# utn = settings.useful_tags_node
# oxna = settings.osm_xml_node_attrs
# oxnt = settings.osm_xml_node_tags
# utw = settings.useful_tags_way
# oxwa = settings.osm_xml_way_attrs
# oxwt = settings.osm_xml_way_tags
# utn = list(set(utn + oxna + oxnt))
# utw = list(set(utw + oxwa + oxwt))
# crs = settings.default_crs
# config(all_oneway=True, useful_tags_node=utn, useful_tags_way=utw, default_crs=crs)
#
# root = et.parse(r"./osm/bj_six_ring.osm").getroot()
# nodes_dict = {}
# for node in root.findall('node'):
#     nodes_dict[int(node.attrib['id'])] = (float(node.attrib['lon']), float(node.attrib['lat']))
#
# M = graph_from_xml('./osm/bj_six_ring.osm', simplify=True)
# # print('G: number_of_nodes:', M.number_of_nodes(), 'number_of_edges:', M.number_of_edges())
#
# # # plot_graph
# # plot_graph(M,
# #             figsize=(16, 16),
# #             bgcolor=None,
# #             node_color="#999999", node_size=15, node_alpha=None, node_edgecolor="none",
# #             edge_color="#999999", edge_linewidth=1, edge_alpha=None,
# #             dpi=600,
# #             save=True, filepath='./data/osm/bj_six_ring.png')
#
# # Convert M to Graph G
# G = nx.Graph()
# G.add_edges_from(Graph(M).edges())
# print("G: number_of_nodes:{0}, number_of_edges:{1}".format(G.number_of_nodes(), G.number_of_edges()))
#
# # write_gexf
# nx.write_gexf(G, 'bj_six_ring.gexf')
#
# # relabel node id
# mapping, index, X = {}, 0, []
# for node in G.nodes:
#     mapping[node] = index
#     lon, lat = nodes_dict[node][0], nodes_dict[node][1]
#     X.append(np.array([lon, lat], dtype=float))
#     index += 1
# X = np.array(X, dtype=float)
# G = nx.relabel_nodes(G, mapping)
# nodes = np.array([nd for nd in G.nodes], dtype=int)

# Add  node attributes
def set_node_attributes(G, X):
    attr = {}
    for node in G.nodes(data=False):
        attr[node] = {'x_'+str(j): X[node, j] for j in range(X.shape[1])}
    nx.set_node_attributes(G, attr)
    return G

def X2Shpfile(X, shpfile = "X_shp.shp", input_crs = "EPSG:4326", output_crs = "EPSG:3857"):
    # input_crs = "EPSG:2436"# input_crs = "EPSG:4326"
    # output_crs = "EPSG:4326"# output_crs = "EPSG:2436"
    # shpfile = "X_shp.shp"
    coords, IDs = [], []
    for i in range(X.shape[0]):
        coords.append(Point(X[i, 0], X[i, 1]))# (X, Y)
        IDs.append("%d" % i)
    data = {'ID': IDs, 'geometry': coords}
    gdf = gpd.GeoDataFrame(data, crs=input_crs)
    # print(gdf.crs)  # 查看数据对应的投影信息(坐标参考系)
    if output_crs is not None:
        gdf.to_crs(crs=output_crs)# (Lon, Lat)
    gdf.to_file(shpfile)

def Edge2Shpfile(G, X, shpfile = "edge_shp.shp", data=True, input_crs = "EPSG:4326", output_crs = "EPSG:3857"):
    # input_crs = "EPSG:2436"  # input_crs = "EPSG:4326"
    # output_crs = "EPSG:4326"  # output_crs = "EPSG:2436"
    # shpfile = "edge_shp.shp"
    coords, weights, edgeIDs = [], [], []
    if data:
        edges = [(u, v, w['weight']) for u, v, w in G.edges(data=data)]
    else:
        edges = [(u, v, 1.0) for u, v in G.edges(data=data)]
    id = 0
    for u, v, w in edges:
        edgeIDs.append(id)
        coords.append((([X[u, 0], X[u, 1]]), ([X[v, 0], X[v, 1]])))
        weights.append(w)
        id += 1
    data = {'edgeID': edgeIDs, 'weight': weights, 'geometry': MultiLineString(coords)}
    gdf = gpd.GeoDataFrame(data, crs=input_crs)
    # print(gdf.crs)  # 查看数据对应的投影信息(坐标参考系)
    if output_crs is not None:
        gdf.to_crs(crs=output_crs)
    gdf.to_file(shpfile)

def LonLat2XY(X, input_crs = "EPSG:4326", output_crs = "EPSG:3857", shpfile_name = "point_shap.shp"):
    # index = ['node_'+str(node) for node in range(X.shape[0])]
    # columns = ['column_'+str(col) for col in range(X.shape[1])]
    # df = pd.DataFrame(X, index=index, columns=columns)

    coords, IDs = [], []
    for i in range(X.shape[0]):
        coords.append(Point(X[i, 0], X[i, 1]))  # (X, Y)
        IDs.append("%d" % i)

    data = {'ID': IDs, 'geometry': coords}
    gdf = gpd.GeoDataFrame(data, crs=input_crs)

    print('input crs', gdf.crs)  # 查看数据对应的投影信息(坐标参考系)
    if output_crs is not None:
        gdf.to_crs(crs=output_crs)  # (Lon, Lat)

    print('output crs', gdf.crs)  # 查看数据对应的投影信息(坐标参考系)  output_crs = "EPSG:3857" or "EPSG:2436"
    if shpfile_name is not None:# shpfile = "point_shap.shp"
        gdf.to_file(filename=shpfile_name)
        print('shpfile is saved')

    geo = gdf.geometry.values
    X0, Y0 = geo.x, geo.y
    X = np.zeros(shape=(X0.shape[0], 2), dtype=float)
    for i in range(X0.shape[0]):
        X[i, 0], X[i, 1] = X0[i], Y0[i]

    return X

def get_G_X_formOSM(osmfile=r'./osm/bj_six_ring.osm', simplify=True, gexf_save=True, plot_graph=False):
    # settings and config
    utn = settings.useful_tags_node
    oxna = settings.osm_xml_node_attrs
    oxnt = settings.osm_xml_node_tags
    utw = settings.useful_tags_way
    oxwa = settings.osm_xml_way_attrs
    oxwt = settings.osm_xml_way_tags
    utn = list(set(utn + oxna + oxnt))
    utw = list(set(utw + oxwa + oxwt))
    crs = settings.default_crs
    config(all_oneway=True, useful_tags_node=utn, useful_tags_way=utw, default_crs=crs)

    root = et.parse(osmfile).getroot()
    nodes_dict = {}
    for node in root.findall('node'):
        nodes_dict[int(node.attrib['id'])] = (float(node.attrib['lon']), float(node.attrib['lat']))

    M = graph_from_xml(osmfile, simplify=simplify)
    # print('G: number_of_nodes:', M.number_of_nodes(), 'number_of_edges:', M.number_of_edges())

    # plot_graph
    if plot_graph:
        plot_graph(M,
                    figsize=(16, 16),
                    bgcolor=None,
                    node_color="#999999", node_size=15, node_alpha=None, node_edgecolor="none",
                    edge_color="#999999", edge_linewidth=1, edge_alpha=None,
                    dpi=600,
                    save=True, filepath='./osm.png')

    # Convert M to Graph G
    G = nx.Graph()
    G.add_edges_from(Graph(M).edges())
    print("G: number_of_nodes:{0}, number_of_edges:{1}".format(G.number_of_nodes(), G.number_of_edges()))

    # write_gexf
    if gexf_save:
        nx.write_gexf(G, 'osm.gexf')

    # relabel node id
    mapping, index, X = {}, 0, []
    for node in G.nodes:
        mapping[node] = index
        lon, lat = nodes_dict[node][0], nodes_dict[node][1]
        X.append(np.array([lon, lat], dtype=float))
        index += 1
    X = np.array(X, dtype=float)
    G = nx.relabel_nodes(G, mapping)
    return G, X

def save_pkl(G, X, X_trans):
    output = open('X_osm.pkl', 'wb')
    pickle.dump(X, output)
    output.close()

    output = open('X_osm_trans.pkl', 'wb')
    pickle.dump(X_trans, output)
    output.close()

    output = open('G_osm.pkl', 'wb')
    pickle.dump(G, output)
    output.close()

def load_GX_pkl():
    output = open('X_osm.pkl', 'rb')
    X = pickle.load(output)
    output.close()

    output = open('X_osm_trans.pkl', 'rb')
    X_trans = pickle.load(output)
    output.close()

    output = open('G_osm.pkl', 'rb')
    G = pickle.load(output)
    output.close()

    return G, X, X_trans

def pkl2txt(di_pkl_name = 'pk1_DI.pkl'):
    output = open(di_pkl_name, 'rb')
    DI = pickle.load(output)
    output.close()
    txt_name = di_pkl_name.split('.')[0]+'.txt'
    file = open(txt_name, mode='w')
    count = 0
    for di in DI:
        str_di = '{0},{1}\n'.format(count, di)
        file.write(str_di)
        count += 1
    file.close()


def get_initial_partition(pk_partion):
    initial_partition = {nd: clus for clus in range(len(pk_partion)) for nd in pk_partion[clus]}
    return initial_partition

# Priori Knowledge 1
def get_pk_MiniBatchKMeans(X, nodes, n_clusters=2, init_size=2, batch_size=500, n_init=10, max_no_improvement=10, verbose=0):
    # # # # # # # # # # # # # # Priori Knowledges # # # # # # # # # # # # # # #
    # Priori Knowledge 1
    # n_clusters = 2542
    mbkm = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters,
                           batch_size=batch_size, n_init=n_init,
                           init_size=init_size,
                           max_no_improvement=max_no_improvement, verbose=verbose).fit(X)
    labels_ = mbkm.labels_
    print("X -> SihCoe score: %0.3f" % metrics.silhouette_score(X, labels_))
    print("X -> CN score: %0.3f" % metrics.calinski_harabasz_score(X, labels_))
    pk_partion = []
    if -1 in labels_:
        index = np.where(labels_ == -1)[0]
        pk_p = nodes[index].tolist()
        for pk in pk_p:
            pk_partion.append([pk])
        n_clusters = len(set(labels_)) - 1
        print('n_clusters', n_clusters)
        for lab in range(0, n_clusters, 1):
            index = np.where(labels_ == lab)[0]
            pk_p = nodes[index].tolist()
            pk_partion.append(pk_p)
    else:
        n_clusters = len(set(labels_))
        for lab in range(0, n_clusters, 1):
            index = np.where(labels_ == lab)[0]
            pk_p = nodes[index].tolist()
            pk_partion.append(pk_p)
    # sort
    results = []
    for sb_result in pk_partion:
        sb_result.sort()
        results.append(sb_result)
    results.sort()
    pk_partion = results

    output = open('mbkm_pk_partion.pkl', 'wb')
    pickle.dump(pk_partion, output)
    output.close()
    return pk_partion

# Priori Knowledge 2
def get_pk_DBSCAN(X_trans, nodes, eps=0.02, min_samples=25):
    # Priori Knowledge 2
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_trans)
    labels_ = db.labels_
    # print("X -> SC score: %0.3f" % metrics.silhouette_score(X_trans, labels_))
    # print("X -> CN score: %0.3f" % metrics.calinski_harabasz_score(X_trans, labels_))
    pk_partion = []
    if -1 in labels_:
        index = np.where(labels_ == -1)[0]
        pk_p = nodes[index].tolist()
        for pk in pk_p:
            pk_partion.append([pk])
        n_clusters = len(set(labels_)) - 1
        for lab in range(0, n_clusters, 1):
            index = np.where(labels_ == lab)[0]
            pk_p = nodes[index].tolist()
            pk_partion.append(pk_p)
    else:
        n_clusters = len(set(labels_))
        print('n_clusters', n_clusters)
        for lab in range(0, n_clusters, 1):
            index = np.where(labels_ == lab)[0]
            pk_p = nodes[index].tolist()
            pk_partion.append(pk_p)
    # sort
    results = []
    for sb_result in pk_partion:
        sb_result.sort()
        results.append(sb_result)
    results.sort()
    pk_partion = results
    # print(pk_partion)
    # print(len(pk_partion))

    output = open('db_pk_partion.pkl', 'wb')
    pickle.dump(pk_partion, output)
    output.close()
    return pk_partion

########################## Test #################################
def main():
    # G, X = get_G_X_formOSM(osmfile=r'../osm/bj_six_ring.osm', simplify=True, gexf_save=True, plot_graph=False)
    # # X_trans
    # X_trans = StandardScaler().fit_transform(X)
    # # save pkl
    # save_pkl(G, X, X_trans)

    G, X, X_trans = load_GX_pkl()

    nodes = np.array([nd for nd in G.nodes], dtype=int)
    nodes_list = nodes.tolist()

    OSIE = get_oneStruInforEntropy(G, weight=None)
    print('oneStruInforEntropy --> OSIE:', OSIE)

    # # # # # # # # # # # # # # Priori Knowledges # # # # # # # # # # # # # # #
    # # Priori Knowledge 1
    # pk_partion1 = get_pk_MiniBatchKMeans(X, nodes=nodes, n_clusters=2542, init_size=2542)
    # DI = get_DI_Parallel(G, partion=pk_partion1, weight=None, n_jobs=4, verbose=0)
    # print('pk_partion1 --> DI:', DI)
    # pk_partion1_hat = get_PKResult(G, initial_partition=get_initial_partition(pk_partion1), gshow=False, edge_data_flg=False)
    # output = open('pk_partion1.pkl', 'wb')
    # pickle.dump(pk_partion1, output)
    # output.close()
    # output = open('pk_partion1_hat.pkl', 'wb')
    # pickle.dump(pk_partion1_hat, output)
    # output.close()

    # Priori Knowledge 2
    pk_partion2 = get_pk_DBSCAN(X_trans=X, nodes=nodes, eps=0.00185, min_samples=2)
    DI = get_DI_Parallel(G, partion=pk_partion2, weight=None, n_jobs=4, verbose=0)
    print('pk_partion2 --> DI:', DI)
    # pk_partion2_hat = get_PKResult(G, initial_partition=get_initial_partition(pk_partion2), gshow=False, edge_data_flg=False)
    # output = open('pk_partion2.pkl', 'wb')
    # pickle.dump(pk_partion2, output)
    # output.close()
    # output = open('pk_partion2_hat.pkl', 'wb')
    # pickle.dump(pk_partion2_hat, output)
    # output.close()

    # Priori Knowledge 3
    pk_partion3 = get_PKResult(G, initial_partition=None, gshow=False, edge_data_flg=False)
    DI = get_DI_Parallel(G, partion=pk_partion3, weight=None, n_jobs=4, verbose=0)
    print('pk_partion3 --> DI:', DI)
    output = open('pk_partion3.pkl', 'wb')
    pickle.dump(pk_partion3, output)
    output.close()

    # # iter_maxDI
    # print('pk_partion1_hat......')
    # results, DI = iter_maxDI(G, iter_max=1000, pk_partion=pk_partion1_hat, weight=None, n_jobs=4, verbose=0)  # 并行加速
    # output = open('pk1_hat_results.pkl', 'wb')
    # pickle.dump(results, output)
    # output.close()
    # output = open('pk1_hat_DI.pkl', 'wb')
    # pickle.dump(DI, output)
    # output.close()

    # # iter_maxDI
    # print('pk_partion2_hat......')
    # results, DI = iter_maxDI(G, iter_max=1000, pk_partion=pk_partion2_hat, weight=None, n_jobs=4, verbose=0)  # 并行加速
    # output = open('pk2_hat_results.pkl', 'wb')
    # pickle.dump(results, output)
    # output.close()
    # output = open('pk2_hat_DI.pkl', 'wb')
    # pickle.dump(DI, output)
    # output.close()

    # # iter_maxDI
    # print('pk_partion1......')
    # results, DI = iter_maxDI(G, iter_max=10000, pk_partion=pk_partion1, weight=None, n_jobs=4, verbose=0)  # 并行加速
    # output = open('pk1_results.pkl', 'wb')
    # pickle.dump(results, output)
    # output.close()
    # output = open('pk1_DI.pkl', 'wb')
    # pickle.dump(DI, output)
    # output.close()
    # pkl2txt(di_pkl_name='pk1_DI.pkl')

    # # iter_maxDI
    # print('pk_partion2......')
    # results, DI = iter_maxDI(G, iter_max=10000, pk_partion=pk_partion2, weight=None, n_jobs=4, verbose=0)  # 并行加速
    # output = open('pk2_results.pkl', 'wb')
    # pickle.dump(results, output)
    # output.close()
    # output = open('pk2_DI.pkl', 'wb')
    # pickle.dump(DI, output)
    # output.close()
    # pkl2txt(di_pkl_name='pk2_DI.pkl')

    # # iter_maxDI
    # print('pk_partion3......')
    # results, DI = iter_maxDI(G, iter_max=1000, pk_partion=pk_partion3, weight=None, n_jobs=4, verbose=0)  # 并行加速
    # output = open('pk3_results.pkl', 'wb')
    # pickle.dump(results, output)
    # output.close()
    # output = open('pk3_DI.pkl', 'wb')
    # pickle.dump(DI, output)
    # output.close()

    # iter_maxDI
    print('nopk_results......')
    results, DI = iter_maxDI(G, iter_max=100000, pk_partion=None, weight=None, n_jobs=4, verbose=0)  # 并行加速
    output = open('nopk_results.pkl', 'wb')
    pickle.dump(results, output)
    output.close()
    output = open('nopk_DI.pkl', 'wb')
    pickle.dump(DI, output)
    output.close()
    pkl2txt(di_pkl_name='nopk_DI.pkl')

if __name__ == '__main__':
    main()
