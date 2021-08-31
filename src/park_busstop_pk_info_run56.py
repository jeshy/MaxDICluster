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

def X_LonLat2XY(X, input_crs = "EPSG:4326", output_crs = "EPSG:3857", shpfile_name = "point_shap.shp"):
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

def load_park_GX_pkl():
    output = open('./park/park_X.pkl', 'rb')
    X = pickle.load(output)
    output.close()

    output = open('./park/park_knnG.pkl', 'rb')
    park_knnG = pickle.load(output)
    output.close()

    output = open('./park/park_radiusG.pkl', 'rb')
    park_radiusG = pickle.load(output)
    output.close()

    return park_knnG, park_radiusG, X

def load_busstop_GX_pkl():
    output = open('./busstop/busstop_X.pkl', 'rb')
    X = pickle.load(output)
    output.close()

    output = open('./busstop/busstop_knnG.pkl', 'rb')
    busstop_knnG = pickle.load(output)
    output.close()

    output = open('./busstop/busstop_radiusG.pkl', 'rb')
    busstop_radiusG = pickle.load(output)
    output.close()

    return busstop_knnG, busstop_radiusG, X

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
    print("X -> SC score: %0.3f" % metrics.silhouette_score(X_trans, labels_))
    print("X -> CN score: %0.3f" % metrics.calinski_harabasz_score(X_trans, labels_))
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

def experiment4(G, str_name, weight=None):
    # G = park_knnG
    # str_name = 'park_knnG'
    # Priori Knowledge 3
    pk_partion = get_PKResult(G, initial_partition=None, gshow=False, edge_data_flg=True)
    DI = get_DI_Parallel(G, partion=pk_partion, weight=weight, n_jobs=4, verbose=0)
    print(str_name+'_pk_partion --> DI:', DI)
    output = open(str_name+'_pk_partion.pkl', 'wb')
    pickle.dump(pk_partion, output)
    output.close()
    # iter_maxDI
    print(str_name+'_pk_partion ......')
    results, DI = iter_maxDI(G, iter_max=200, pk_partion=pk_partion, weight=weight, n_jobs=4, verbose=0)  # 并行加速
    output = open(str_name+'_pk_results.pkl', 'wb')
    pickle.dump(results, output)
    output.close()
    output = open(str_name+'_pk_DI.pkl', 'wb')
    pickle.dump(DI, output)
    output.close()
    print(str_name+' is ok')

def experiment5(G, X, str_name='', n_clusters=None):
    # G, X, str_name = park_knnG, park_X, 'park_knnG'
    OSIE = get_oneStruInforEntropy(G, weight='weight')

    nodes = np.array([nd for nd in G.nodes], dtype=int)
    if n_clusters is None:
        n_clusters = len(get_PKResult(G, initial_partition=None, gshow=False, edge_data_flg=True))
    print('size_0:', n_clusters)

    pk_partion = get_pk_MiniBatchKMeans(X, nodes=nodes, n_clusters=n_clusters, init_size=n_clusters)

    DI = get_DI_Parallel(G, partion=pk_partion, weight='weight', n_jobs=4, verbose=0)
    R = DI / OSIE
    print(str_name+'_R0: %.3f' % R)

    print('------maxDI-------')

    pk_partion = get_PKResult(G, initial_partition=get_initial_partition(pk_partion), gshow=False, edge_data_flg=True)
    results, DI = iter_maxDI(G, iter_max=100, pk_partion=pk_partion, weight='weight', n_jobs=4, verbose=0)  # 并行加速
    R = DI[-1] / OSIE
    print('size:', len(results))
    print(str_name+'_R1: %.3f' % R)

    labels_ = label_result(nsamples=X.shape[0], result=results).astype(int)
    print("X -> SihCoe score: %0.3f" % metrics.silhouette_score(X, labels_))
    print("X -> CN score: %0.3f" % metrics.calinski_harabasz_score(X, labels_))

def experiment5_1(G, X, str_name='', eps=500, min_samples=2):
    # G, X, str_name = park_knnG, park_X, 'park_knnG'
    OSIE = get_oneStruInforEntropy(G, weight='weight')

    nodes = np.array([nd for nd in G.nodes], dtype=int)
    pk_partion = get_pk_DBSCAN(X_trans=X, nodes=nodes, eps=eps, min_samples=min_samples)

    DI = get_DI_Parallel(G, partion=pk_partion, weight='weight', n_jobs=4, verbose=0)
    R = DI / OSIE
    print(str_name+'_R0: %.3f' % R)

    print('------maxDI-------')

    pk_partion = get_PKResult(G, initial_partition=get_initial_partition(pk_partion), gshow=False, edge_data_flg=True)
    results, DI = iter_maxDI(G, iter_max=100, pk_partion=pk_partion, weight='weight', n_jobs=4, verbose=0)  # 并行加速
    R = DI[-1] / OSIE
    print('size:', len(results))
    print(str_name+'_R1: %.3f' % R)

    labels_ = label_result(nsamples=X.shape[0], result=results).astype(int)
    print("X -> SihCoe score: %0.3f" % metrics.silhouette_score(X, labels_))
    print("X -> CN score: %0.3f" % metrics.calinski_harabasz_score(X, labels_))


########################## Test #################################
def main():
    park_knnG, park_radiusG, park_X = load_park_GX_pkl()
    busstop_knnG, busstop_radiusG, busstop_X = load_busstop_GX_pkl()

    print('------------------------------OSIE----------------------------------')
    # A. OSIE
    OSIE = get_oneStruInforEntropy(park_knnG, weight='weight')
    print("park_knnG: number_of_nodes:{0}, number_of_edges:{1}".format(park_knnG.number_of_nodes(), park_knnG.number_of_edges()))
    print('park_knnG --> OSIE:%.3f' % OSIE)
    OSIE = get_oneStruInforEntropy(park_radiusG, weight='weight')
    print("park_radiusG: number_of_nodes:{0}, number_of_edges:{1}".format(park_radiusG.number_of_nodes(), park_radiusG.number_of_edges()))
    print('park_radiusG --> OSIE:%.3f' % OSIE)
    # B. OSIE
    OSIE = get_oneStruInforEntropy(busstop_knnG, weight='weight')
    print("busstop_knnG: number_of_nodes:{0}, number_of_edges:{1}".format(busstop_knnG.number_of_nodes(), busstop_knnG.number_of_edges()))
    print('busstop_knnG --> OSIE:%.3f' % OSIE)
    OSIE = get_oneStruInforEntropy(busstop_radiusG, weight='weight')
    print("busstop_radiusG: number_of_nodes:{0}, number_of_edges:{1}".format(busstop_radiusG.number_of_nodes(), busstop_radiusG.number_of_edges()))
    print('busstop_radiusG --> OSIE:%.3f' % OSIE)
    print('------------------------------OSIE-end---------------------------------')

    print('------------------------------experiment4----------------------------------')
    print('*******park_knnG*******')
    experiment4(G=park_knnG, str_name='park_knnG', weight='weight')
    print('*******park_radiusG*******')
    experiment4(G=park_radiusG, str_name='park_radiusG', weight='weight')

    print('*******busstop_knnG*******')
    experiment4(G=busstop_knnG, str_name='busstop_knnG', weight='weight')
    print('*******busstop_radiusG*******')
    experiment4(G=busstop_radiusG, str_name='busstop_radiusG', weight='weight')
    print('------------------------------experiment4-end---------------------------------')

    print('------------------------------experiment5----------------------------------')
    print('******* KM - park_knnG *******')
    experiment5(G=park_knnG, X=park_X, str_name='park_knnG', n_clusters=120)
    print('******* KM - park_radiusG *******')
    experiment5(G=park_radiusG, X=park_X, str_name='park_radiusG', n_clusters=120)

    print('******* DB - park_knnG *******')
    experiment5_1(G=park_knnG, X=park_X, str_name='park_knnG', eps=500, min_samples=2)
    print('******* DB - park_radiusG *******')
    experiment5_1(G=park_radiusG, X=park_X, str_name='park_radiusG', eps=500, min_samples=2)

    print('******* KM - busstop_knnG *******')
    experiment5(G=busstop_knnG, X=busstop_X, str_name='busstop_knnG', n_clusters=450)
    print('******* KM - busstop_radiusG *******')
    experiment5(G=busstop_radiusG, X=busstop_X, str_name='busstop_radiusG', n_clusters=450)

    print('******* DB - busstop_knnG *******')
    experiment5_1(G=busstop_knnG, X=busstop_X, str_name='busstop_knnG', eps=500, min_samples=2)
    print('******* DB - busstop_radiusG *******')
    experiment5_1(G=busstop_radiusG, X=busstop_X, str_name='busstop_radiusG', eps=500, min_samples=2)

    print('------------------------------experiment5-end---------------------------------')

    # ------------------------------OSIE----------------------------------
    # park_knnG: number_of_nodes:5881, number_of_edges:23758
    # park_knnG --> OSIE:8.920
    # park_radiusG: number_of_nodes:5881, number_of_edges:28590
    # park_radiusG --> OSIE:7.834
    # busstop_knnG: number_of_nodes:42161, number_of_edges:513377
    # busstop_knnG --> OSIE:7.199
    # busstop_radiusG: number_of_nodes:42161, number_of_edges:516241
    # busstop_radiusG --> OSIE:7.433
    # ------------------------------OSIE-end---------------------------------
    # ------------------------------experiment4----------------------------------
    # *******park_knnG*******
    # park_knnG_pk_partion --> DI: 6.969
    # park_knnG_pk_partion ......
    # (iter:0 ---> DI:6.969 bits)
    # (iter:50 ---> DI:6.969 bits)
    # (iter:100 ---> DI:6.968 bits)
    # (iter:150 ---> DI:6.968 bits)
    # (iter:200 ---> DI:6.968 bits)
    # park_knnGis ok
    # *******park_radiusG*******
    # park_radiusG_pk_partion --> DI: 5.921
    # park_radiusG_pk_partion ......
    # (iter:0 ---> DI:5.921 bits)
    # (iter:50 ---> DI:5.921 bits)
    # (iter:100 ---> DI:5.921 bits)
    # (iter:150 ---> DI:5.921 bits)
    # (iter:200 ---> DI:5.921 bits)
    # park_radiusGis ok
    # *******busstop_knnG*******
    # busstop_knnG_pk_partion --> DI: 6.001
    # busstop_knnG_pk_partion ......
    # (iter:0 ---> DI:6.001 bits)
    # (iter:50 ---> DI:6.000 bits)
    # (iter:100 ---> DI:5.993 bits)
    # (iter:150 ---> DI:5.988 bits)
    # (iter:200 ---> DI:5.987 bits)
    # busstop_knnGis ok
    # *******busstop_radiusG*******
    # busstop_radiusG_pk_partion --> DI: 6.264
    # busstop_radiusG_pk_partion ......
    # (iter:0 ---> DI:6.264 bits)
    # (iter:50 ---> DI:6.264 bits)
    # (iter:100 ---> DI:6.256 bits)
    # (iter:150 ---> DI:6.253 bits)
    # (iter:200 ---> DI:6.252 bits)
    # busstop_radiusGis ok
    # ------------------------------experiment4-end---------------------------------
    # ------------------------------experiment5----------------------------------
    # ******* KM - park_knnG *******
    # size_0: 120
    # X -> SihCoe score: 0.465
    # X -> CN score: 2647.645
    # park_knnG_R0: 0.651
    # ------maxDI-------
    # (iter:0 ---> DI:6.947 bits)
    # (iter:50 ---> DI:6.947 bits)
    # (iter:100 ---> DI:6.947 bits)
    # size: 336
    # park_knnG_R1: 0.779
    # X -> SihCoe score: 0.112
    # X -> CN score: 302.085
    # ******* KM - park_radiusG *******
    # size_0: 120
    # X -> SihCoe score: 0.455
    # X -> CN score: 3021.169
    # park_radiusG_R0: 0.599
    # ------maxDI-------
    # (iter:0 ---> DI:5.921 bits)
    # (iter:50 ---> DI:5.921 bits)
    # (iter:100 ---> DI:5.921 bits)
    # size: 264
    # park_radiusG_R1: 0.756
    # X -> SihCoe score: 0.231
    # X -> CN score: 220.516
    # ******* DB - park_knnG *******
    # X -> SC score: -0.255
    # X -> CN score: 44.614
    # park_knnG_R0: 0.553
    # ------maxDI-------
    # (iter:0 ---> DI:6.947 bits)
    # (iter:50 ---> DI:6.947 bits)
    # (iter:100 ---> DI:6.947 bits)
    # size: 332
    # park_knnG_R1: 0.779
    # X -> SihCoe score: 0.071
    # X -> CN score: 260.957
    # ******* DB - park_radiusG *******
    # X -> SC score: -0.255
    # X -> CN score: 44.614
    # park_radiusG_R0: 0.607
    # ------maxDI-------
    # (iter:0 ---> DI:5.921 bits)
    # (iter:50 ---> DI:5.921 bits)
    # (iter:100 ---> DI:5.921 bits)
    # size: 270
    # park_radiusG_R1: 0.756
    # X -> SihCoe score: 0.255
    # X -> CN score: 241.418
    # ******* KM - busstop_knnG *******
    # size_0: 450
    # X -> SihCoe score: 0.387
    # X -> CN score: 33173.224
    # busstop_knnG_R0: 0.781
    # ------maxDI-------
    # (iter:0 ---> DI:6.001 bits)
    # (iter:50 ---> DI:6.000 bits)
    # (iter:100 ---> DI:5.993 bits)
    # size: 3945
    # busstop_knnG_R1: 0.832
    # X -> SihCoe score: 0.497
    # X -> CN score: 1617.800
    # ******* KM - busstop_radiusG *******
    # size_0: 450
    # X -> SihCoe score: 0.385
    # X -> CN score: 27508.975
    # busstop_radiusG_R0: 0.697
    # (iter:0 ---> DI:6.262 bits)
    # (iter:50 ---> DI:6.262 bits)
    # (iter:100 ---> DI:6.254 bits)
    # size: 3951
    # busstop_radiusG_R1: 0.841
    # X -> SihCoe score: 0.495
    # X -> CN score: 1511.164
    # ******* DB - busstop_knnG *******
    # X -> SC score: -0.147
    # X -> CN score: 161.861
    # busstop_knnG_R0: 0.622
    # (iter:0 ---> DI:6.001 bits)
    # (iter:50 ---> DI:5.997 bits)
    # (iter:100 ---> DI:5.989 bits)
    # size: 3914
    # busstop_knnG_R1: 0.832
    # X -> SihCoe score: 0.491
    # X -> CN score: 1672.600
    # ******* DB - busstop_radiusG *******
    # (iter:50 ---> DI:6.262 bits)
    # (iter:100 ---> DI:6.254 bits)
    # size: 3952
    # busstop_radiusG_R1: 0.841
    # X -> SihCoe score: 0.491
    # X -> CN score: 1579.694
    # ------------------------------experiment5-end---------------------------------

if __name__ == '__main__':
    main()
