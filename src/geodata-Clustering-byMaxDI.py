import pickle
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import xml.etree.cElementTree as et
from shapely.geometry import Point, MultiLineString, Polygon
from sklearn.metrics.pairwise import pairwise_kernels, rbf_kernel
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from utils_SI import plot_graph, get_DI_Parallel, label_result, iter_maxDI, get_oneStruInforEntropy
# utils_SI：https://www.cnblogs.com/jeshy/p/15032483.html

def XDecomposition(X, n_components=2):
    from sklearn.decomposition import PCA
    (m, n) = X.shape
    if n_components >= n:
        return X
    else:
        pca = PCA(n_components=n_components, svd_solver='full')
        pca.fit(X)
        X_trans = pca.transform(X)
        # pca.explained_variance_ratio_
        # pca.singular_values_
        return X_trans

# Add  node attributes
def set_node_attributes(G, X):
    attr = {}
    for node in G.nodes(data=False):
        attr[node] = {'x_'+str(j): X[node, j] for j in range(X.shape[1])}
    nx.set_node_attributes(G, attr)
    return G

# Add  part_nodes attributes
def set_part_node_attributes(G, nodelist, X):
    attr = {}
    for node in nodelist:
        attr[node] = {'x_'+str(j): X[node, j] for j in range(X.shape[1])}
    nx.set_node_attributes(G, attr)
    return G

def update_node_attributes(G, X, node_new):
    attr = {}
    attr[node_new] = {'x_' + str(j): X[node_new, j] for j in range(X.shape[1])}
    nx.set_node_attributes(G, attr)
    return G

def query_GraphND_cKDTree(G, points, num_feature):
    # X --> m x n; points  --> k x n
    # points = np.array([X, Y]).T
    nodes = {'x_' + str(j): nx.get_node_attributes(G, 'x_' + str(j)) for j in range(num_feature)}
    nodes_pd = pd.DataFrame(nodes)
    data = nodes_pd[['x_'+str(j) for j in range(num_feature)]]

    tree = cKDTree(data=data, compact_nodes=True, balanced_tree=True)

    dist, idx = tree.query(points, k=1)
    node = nodes_pd.iloc[idx].index

    return node, dist

def get_poly_from_osm_circleline(circleline_osmfile=r"../osm/beijing_car_seg_6ring road.osm", toepsg=2436):# toepsg == 3857; 2436
    # 精确的北京六环道路 见 beijing_car_seg_6ring road.osm： （北京六环附近及往内的可驾驶道路网络（路网graph为连通图））https://www.cnblogs.com/jeshy/p/14763489.html
    # beijing_car_seg_6ring road.osm as the input data
    # root = et.parse(r"./beijing_car_seg_6ring road.osm").getroot()
    root = et.parse(circleline_osmfile).getroot()
    nodes = {}
    for node in root.findall('node'):
        nodes[int(node.attrib['id'])] = (float(node.attrib['lon']), float(node.attrib['lat']))
    edges = []
    for way in root.findall('way'):
        element_nd = way.findall('nd')
        if len(element_nd) > 2:
            for i in range(len(element_nd) - 1):
                node_s, node_e = int(element_nd[i].attrib['ref']), int(element_nd[i+1].attrib['ref'])
                path = (node_s, node_e)
                edges.append(path)
        else:
            node_s, node_e = int(element_nd[0].attrib['ref']), int(element_nd[1].attrib['ref'])
            path = (node_s, node_e)
            edges.append(path)
    edge = edges[0]
    E = []
    E.append(edge)
    edges.remove(edge)
    point_s, point_e = nodes[E[0][0]], nodes[E[0][1]]
    Point_lists = []
    Point_lists.append(list(point_s))
    Point_lists.append(list(point_e))
    while len(edges) > 0:
        (node_f_s, node_f_e) = E[-1]
        for (node_n_s, node_n_e) in edges:
            if node_f_e == node_n_s:
                E.append((node_n_s, node_n_e))
                point_f_e = nodes[node_n_e]
                Point_lists.append(list(point_f_e))
                # print((node_n_s, node_n_e))
                edges.remove((node_n_s, node_n_e))
                break
    # Points.pop()
    # print(E[0], '...', E[-2], E[-1])
    # print(Point_lists[0], '...', Point_lists[-2], Point_lists[-1])

    road_line_arr = np.array(Point_lists)  # 转换成二维坐标表示
    sixc_ring_poly = Polygon(road_line_arr)  # Polygon

    crs = {'init': 'epsg:4326'}
    gpoly = gpd.GeoSeries(sixc_ring_poly, crs=crs)

    if toepsg is not None:
        gpoly = gpoly.to_crs(epsg=toepsg)
    print('output gpoly.crs:', gpoly.crs)

    # poly = gpoly[0]
    # print('area(km*km):', poly.area / 1.0e6)
    # print('length(km):', poly.length / 1.0e3)

    return road_line_arr, sixc_ring_poly, gpoly

def X2GDF(X, input_crs = "EPSG:2436", output_crs = "EPSG:4326", plot=True):
    # input_crs = "EPSG:2436"# input_crs = "EPSG:4326"
    # output_crs = "EPSG:4326"# output_crs = "EPSG:2436"
    coords, IDs = [], []
    for i in range(X.shape[0]):
        coords.append(Point(X[i, 0], X[i, 1]))# (X, Y)
        IDs.append("%d" % i)
    data = {'ID': IDs, 'geometry': coords}
    gdf = gpd.GeoDataFrame(data, crs=input_crs)
    # print(gdf.crs)  # 查看数据对应的投影信息(坐标参考系)
    if output_crs is not None:
        gdf.to_crs(crs=output_crs)# (Lon, Lat)

    if plot:
        gdf.plot()
        plt.show()#展示

    return gdf

def GDF2X(gdata, to_epsg=2436, plot=True):
    print('input gdf crs:', gdata.crs)  # 查看数据对应的投影信息(坐标参考系) ---> epsg == 4326  (WGS_84)
    gdata = gdata.to_crs(epsg=to_epsg)# epsg == 2436 (X, Y), 4326 (lon, lat)
    print('output X crs:', gdata.crs)
    # print(gdata.columns.values)# 列名
    # print(gdata.head())  # 查看前5行数据
    if plot:
        gdata.plot()
        plt.show()#展示
    geo = gdata.geometry.values
    X0, Y0 = geo.x, geo.y
    X = np.zeros(shape=(X0.shape[0], 2), dtype=float)
    for i in range(X0.shape[0]):
        X[i, 0], X[i, 1] = X0[i],  Y0[i]
    return X

def LonLat2XY(X, input_crs = "EPSG:4326", output_crs = "EPSG:3857", shpfile_name = "point_shp.shp"):
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

def get_partion_result_outline(X, partion_result, input_crs = "EPSG:2436", output_crs = None, shpfile = "partion_outline_shp.shp", save_shp = True, plot = True):
    import alphashape
    # input_crs = "EPSG:2436"
    # output_crs = None
    # shpfile = "partion_outline_shp.shp"
    # save_shp = True
    # plot = True
    Polygs, Numbers, IDs = [], [], []
    for i in range(len(partion_result)):
        X_uniques = np.unique(X[partion_result[i]], axis=0)# 去除重复行
        if X_uniques.shape[0] > 2:
            gdf_alpha_shape = alphashape.alphashape(X2GDF(X_uniques, input_crs=input_crs, output_crs=output_crs, plot=None))
            polyg = gdf_alpha_shape.geometry.values# GeoDataFrameArray
            Polygs.append(polyg[0])
            Numbers.append(len(partion_result[i]))
            IDs.append("%d" % i)
    data = {'ID': IDs, 'num': Numbers, 'geometry': Polygs}
    gdf = gpd.GeoDataFrame(data, crs=input_crs)

    if output_crs is not None:
        gdf.to_crs(crs=output_crs)

    if save_shp:
        gdf.to_file(shpfile)

    if plot:
        gdf.plot()
        plt.show()  # 展示

    return gdf

def X2Shpfile(X, shpfile="X_shp.shp", input_crs="EPSG:2436", output_crs="EPSG:4326"):
    # input_crs = "EPSG:2436"# input_crs = "EPSG:4326"
    # output_crs = "EPSG:4326"# output_crs = "EPSG:2436"
    # shpfile = "X_shp.shp"
    coords, IDs = [], []
    for i in range(X.shape[0]):
        coords.append(Point(X[i, 0], X[i, 1]))  # (X, Y)
        IDs.append("%d" % i)
    data = {'ID': IDs, 'geometry': coords}
    gdf = gpd.GeoDataFrame(data, crs=input_crs)
    # print(gdf.crs)  # 查看数据对应的投影信息(坐标参考系)
    if output_crs is not None:
        gdf.to_crs(crs=output_crs)  # (Lon, Lat)
    gdf.to_file(shpfile)

def Edge2Shpfile(G, X, shpfile="edge_shp.shp", data=True, input_crs="EPSG:2436", output_crs="EPSG:4326"):
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

def FixingGraph(G, X, merage_components=False):
    # Add  node attributes
    G = set_node_attributes(G, X)
    node_index = [nd for nd in G.nodes]
    if G.number_of_nodes() < X.shape[0]:
        print("Graph Fixing (Connecting the Isolated Points by distances)......... ")
        # Fixed Graph
        node_all_index = [nd for nd in range(X.shape[0])]
        node_non_index = list(set(node_all_index) - set(node_index))
        for ndi in node_non_index:
            ptq = X[ndi:ndi + 1, :]
            indices, distances = query_GraphND_cKDTree(G, points=ptq, num_feature=X.shape[1])
            add_edge = [(ndi, indices[0], {'weight': 1.0 / (distances[0] + 1e-5)})]
            G.add_edges_from(add_edge)
            # update the new node attributes
            G = update_node_attributes(G, X, node_new=ndi)
            node_index.append(ndi)
        print("G(Fixed): number_of_nodes:{0}, number_of_edges:{1}".format(G.number_of_nodes(), G.number_of_edges()))
    # # onnected components in G
    if merage_components:
        connected_components = [list(c) for c in sorted(nx.connected_components(G), key=len, reverse=False)]
        Len_C = len(connected_components)
        print('Graph Components Numbers: ', Len_C)
        print("Graph Components Connecting ......... ")
        while Len_C >= 2:
            first_components = connected_components[0]
            # first_sub_graph
            first_sub_graph = G.subgraph(first_components).copy()

            # remove_nodes_from first_sub_graph for G
            G.remove_nodes_from(first_components)

            index_v = [nd for nd, data in first_sub_graph.nodes(data=True)]
            points = X[index_v, :]

            index_us, distances = query_GraphND_cKDTree(G, points=points, num_feature=X.shape[1])

            index_us, distances = list(index_us), list(distances)
            min_index = distances.index(min(distances))  # 返回最小值 index
            v, u, dis = index_v[min_index], index_us[min_index], distances[min_index]
            bridge_edge = [(v, u, {'weight': 1.0 / (dis + 1e-5)})]
            G.add_edges_from(bridge_edge)

            # update the new node attributes
            G = update_node_attributes(G, X, node_new=v)

            # add the subgraph to G
            sub_graph_edges = [edge for edge in first_sub_graph.edges(data=True)]
            G.add_edges_from(sub_graph_edges)
            nodelist = [node for node in G.nodes(data=False)]
            G = set_part_node_attributes(G, nodelist, X)

            connected_components = [list(c) for c in sorted(nx.connected_components(G), key=len, reverse=False)]
            Len_C = len(connected_components)
        print("G(Finally): number_of_nodes:{0}, number_of_edges:{1}".format(G.number_of_nodes(), G.number_of_edges()))

    return G

def getKNNGraph(X, n_neighbors=10, threshold=2.25, merage_components=False, save=True, shpfile=False):
    # threshold = threshold  ########### setup parameter ###########
    if save:
        output = open('X.pkl', 'wb')
        pickle.dump(X, output)
        output.close()

        # from sklearn.preprocessing import StandardScaler
        # X_trans = StandardScaler().fit_transform(X)
        # output = open('trans_X.pkl', 'wb')
        # pickle.dump(X_trans, output)
        # output.close()
        print("X is saved")

    knn_graph = kneighbors_graph(X, n_neighbors=n_neighbors, mode='distance', include_self=False, n_jobs=-1)
    # knn_graph = kneighbors_graph(X_trans, n_neighbors=n_neighbors, mode='distance', include_self=False, n_jobs=8)
    graph = knn_graph
    affinity = 0.5 * (graph + graph.T)
    edges_list, node_index = [], []
    G = nx.Graph()
    if hasattr(affinity, 'tocoo'):
        A = affinity.tocoo()  # csr_matrix --> coo
        # edges_list = [(i, j, {'weight': 1.0/v}) for i, j, v in zip(A.row, A.col, A.data)]  # coo matrix
        for i, j, v in zip(A.row, A.col, A.data):
            edge = (i, j, {'weight': 1.0 / (v + 1e-5)})
            edges_list.append(edge)
            node_index.append(i)
            node_index.append(j)
    G.add_edges_from(edges_list)
    node_index = list(set(node_index))
    print("knnG(all kneighbors ): number_of_nodes:{0}, number_of_edges:{1}".format(G.number_of_nodes(), G.number_of_edges()))

    print("Graph Edge Removing ......... ")
    for node in node_index:
        neighbors = [nd for nd in nx.neighbors(G, node)]
        if len(neighbors) > 1:
            X_i = X[node, :]
            dis_dict = {}
            for nd in neighbors:
                delt_X_k = X_i - X[nd, :]
                dis = np.linalg.norm(delt_X_k)
                dis_dict[nd] = dis
            dis_order = sorted(dis_dict.items(), key=lambda x: x[1], reverse=False)  # sort by ascending
            (min_id, min_dis), (max_id, max_dis) = dis_order[0], dis_order[-1]
            if threshold >= min_dis and max_dis >= threshold:
                for nd in neighbors:
                    if dis_dict[nd] > threshold:  # remove
                        rm_edge = [(node, nd)]
                        G.remove_edges_from(rm_edge)
            elif threshold < min_dis:
                for nd in neighbors:  # remove all
                    rm_edge = [(node, nd)]
                    G.remove_edges_from(rm_edge)
                add_edge = [(node, nd, {'weight': 1.0 / min_dis})]  # add the small
                G.add_edges_from(add_edge)
            else:  # threshold >= max_dis
                pass  # save all
    print("knnG(edgermove): number_of_nodes:{0}, number_of_edges:{1}".format(G.number_of_nodes(), G.number_of_edges()))

    # Add  node attributes
    G = set_node_attributes(G, X)
    if G.number_of_nodes() < X.shape[0]:
        print("Graph Fixing ......... ")
        # Fixed Graph
        node_all_index = [nd for nd in range(X.shape[0])]
        node_non_index = list(set(node_all_index) - set(node_index))
        for ndi in node_non_index:
            ptq = X[ndi:ndi+1, :]
            indices, distances = query_GraphND_cKDTree(G, points=ptq, num_feature=X.shape[1])
            add_edge = [(ndi, indices[0], {'weight': 1.0 / (distances[0] + 1e-5)})]
            G.add_edges_from(add_edge)
            # update the new node attributes
            G = update_node_attributes(G, X, node_new=ndi)
            node_index.append(ndi)
        print("knnG(Fixed): number_of_nodes:{0}, number_of_edges:{1}".format(G.number_of_nodes(), G.number_of_edges()))
        # test
        # Edge2Shpfile(G, X_pca, shpfile="edge_shp.shp", data=True, input_crs="EPSG:2436", output_crs=None)
        # X2Shpfile(X_pca, shpfile="X_shp.shp", input_crs="EPSG:2436", output_crs=None)
        # print("Shpfile is saved")
    # # onnected components in G
    if merage_components:
        connected_components = [list(c) for c in sorted(nx.connected_components(G), key=len, reverse=False)]
        Len_C = len(connected_components)
        print('Graph Components Numbers: ', Len_C)
        print("Graph Components Connecting ......... ")
        while Len_C >= 2:
            first_components = connected_components[0]
            # first_sub_graph
            first_sub_graph = G.subgraph(first_components).copy()

            # remove_nodes_from first_sub_graph for G
            G.remove_nodes_from(first_components)

            index_v = [nd for nd, data in first_sub_graph.nodes(data=True)]
            points = X[index_v, :]

            index_us, distances = query_GraphND_cKDTree(G, points=points, num_feature=X.shape[1])

            index_us, distances = list(index_us), list(distances)
            min_index = distances.index(min(distances))# 返回最小值 index
            v, u, dis = index_v[min_index], index_us[min_index], distances[min_index]
            bridge_edge = [(v, u, {'weight': 1.0 / (dis + 1e-5)})]
            G.add_edges_from(bridge_edge)

            # update the new node attributes
            G = update_node_attributes(G, X, node_new=v)

            # add the subgraph to G
            sub_graph_edges = [edge for edge in first_sub_graph.edges(data=True)]
            G.add_edges_from(sub_graph_edges)
            nodelist = [node for node in G.nodes(data=False)]
            G = set_part_node_attributes(G, nodelist, X)

            connected_components = [list(c) for c in sorted(nx.connected_components(G), key=len, reverse=False)]
            Len_C = len(connected_components)
        print("knnG(Finally): number_of_nodes:{0}, number_of_edges:{1}".format(G.number_of_nodes(), G.number_of_edges()))

    if save:
        output = open('knnG.pkl', 'wb')
        pickle.dump(G, output)
        output.close()
        print("knnG is saved")

    if shpfile:
        # X Decomposition
        X_pca = XDecomposition(X, n_components=2)
        Edge2Shpfile(G, X_pca, shpfile="edge_shp.shp", data=True, input_crs="EPSG:2436", output_crs=None)
        X2Shpfile(X_pca, shpfile="X_shp.shp", input_crs="EPSG:2436", output_crs=None)
        print("Shpfile is saved")

    return G

def getRadiusGraph(X, radius=2.25, merage_components=False, save=True, shpfile=False):
    # threshold = threshold  ########### setup parameter ###########
    if save:
        output = open('X.pkl', 'wb')
        pickle.dump(X, output)
        output.close()
        # from sklearn.preprocessing import StandardScaler
        # X_trans = StandardScaler().fit_transform(X)
        # output = open('trans_X.pkl', 'wb')
        # pickle.dump(X_trans, output)
        # output.close()
        print("X is saved")

    radius_graph = radius_neighbors_graph(X, radius=radius, mode='distance', include_self=False, n_jobs=-1)
    graph = radius_graph
    affinity = 0.5 * (graph + graph.T)
    edges_list, node_index = [], []
    G = nx.Graph()
    if hasattr(affinity, 'tocoo'):
        A = affinity.tocoo()  # csr_matrix --> coo
        # edges_list = [(i, j, {'weight': 1.0/v}) for i, j, v in zip(A.row, A.col, A.data)]  # coo matrix
        for i, j, v in zip(A.row, A.col, A.data):
            edge = (i, j, {'weight': 1.0 / (v + 1e-5)})
            edges_list.append(edge)
            node_index.append(i)
            node_index.append(j)
    G.add_edges_from(edges_list)
    node_index = list(set(node_index))
    print("radiusG(all radius neighbors ): number_of_nodes:{0}, number_of_edges:{1}".format(G.number_of_nodes(), G.number_of_edges()))

    # Add  node attributes
    G = set_node_attributes(G, X)
    if G.number_of_nodes() < X.shape[0]:
        print("Graph Fixing ......... ")
        # Fixed Graph
        node_all_index = [nd for nd in range(X.shape[0])]
        node_non_index = list(set(node_all_index) - set(node_index))
        for ndi in node_non_index:
            ptq = X[ndi:ndi+1, :]
            indices, distances = query_GraphND_cKDTree(G, points=ptq, num_feature=X.shape[1])
            add_edge = [(ndi, indices[0], {'weight': 1.0 / (distances[0] + 1e-5)})]
            G.add_edges_from(add_edge)
            # update the new node attributes
            G = update_node_attributes(G, X, node_new=ndi)
            node_index.append(ndi)
        print("radiusG(Fixed): number_of_nodes:{0}, number_of_edges:{1}".format(G.number_of_nodes(), G.number_of_edges()))
        # Edge2Shpfile(G, X_pca, shpfile="edge_shp.shp", data=True, input_crs="EPSG:2436", output_crs=None)
        # X2Shpfile(X_pca, shpfile="X_shp.shp", input_crs="EPSG:2436", output_crs=None)
        # print("Shpfile is saved")

    # # onnected components in G
    if merage_components:
        connected_components = [list(c) for c in sorted(nx.connected_components(G), key=len, reverse=False)]
        Len_C = len(connected_components)
        print('Graph Components Numbers: ', Len_C)
        print("Graph Components Connecting ......... ")
        while Len_C >= 2:
            first_components = connected_components[0]
            # first_sub_graph
            first_sub_graph = G.subgraph(first_components).copy()

            # remove_nodes_from first_sub_graph for G
            G.remove_nodes_from(first_components)

            index_v = [nd for nd, data in first_sub_graph.nodes(data=True)]
            points = X[index_v, :]

            index_us, distances = query_GraphND_cKDTree(G, points=points, num_feature=X.shape[1])

            index_us, distances = list(index_us), list(distances)
            min_index = distances.index(min(distances))# 返回最小值 index
            v, u, dis = index_v[min_index], index_us[min_index], distances[min_index]
            bridge_edge = [(v, u, {'weight': 1.0 / (dis + 1e-5)})]
            G.add_edges_from(bridge_edge)

            # update the new node attributes
            G = update_node_attributes(G, X, node_new=v)

            # add the subgraph to G
            sub_graph_edges = [edge for edge in first_sub_graph.edges(data=True)]
            G.add_edges_from(sub_graph_edges)
            nodelist = [node for node in G.nodes(data=False)]
            G = set_part_node_attributes(G, nodelist, X)

            connected_components = [list(c) for c in sorted(nx.connected_components(G), key=len, reverse=False)]
            Len_C = len(connected_components)
        print("radiusG(Finally): number_of_nodes:{0}, number_of_edges:{1}".format(G.number_of_nodes(), G.number_of_edges()))

    if save:
        output = open('radiusG.pkl', 'wb')
        pickle.dump(G, output)
        output.close()
        print("radiusG is saved")

    if shpfile:
        # X Decomposition
        X_pca = XDecomposition(X, n_components=2)
        Edge2Shpfile(G, X_pca, shpfile="edge_shp.shp", data=True, input_crs="EPSG:2436", output_crs=None)
        X2Shpfile(X_pca, shpfile="X_shp.shp", input_crs="EPSG:2436", output_crs=None)
        print("Shpfile is saved")

    return G

def getRBFKernelsGraph(X, gamma=1., threshold=5., merage_components=False, n_jobs=-1, shpfile=False):
    params = {}
    params['gama'] = gamma
    K = pairwise_kernels(X, metric='rbf', filter_params=True, n_jobs=n_jobs, **params)  # A kernel matrix K
    # K.tocoo()
    Kcsr = sp.csr_matrix(K)
    K = Kcsr.tocoo()
    print('A kernel matrix K: ', K.shape)
    # K(x, y) = exp(-gamma ||x-y||^2) ----> gamma = sigma^-2 ; sigma^2 is the Gaussian kernel of variance.
    affinity = 0.5 * (K + K.T)
    edges_list, node_index = [], []
    if hasattr(affinity, 'tocoo'):
        A = affinity.tocoo()
        # edges_list = [(i, j, {'weight': (v + 1e-5)}) for i, j, v in zip(A.row, A.col, A.data) if
        #               v >= threshold and i < j]  # coo matrix
        for i, j, v in zip(A.row, A.col, A.data):  # coo matrix
            if v >= threshold and i < j:
                edges_list.append((i, j, {'weight': v}))
                node_index.append(i)
                node_index.append(j)
    else:
        A = affinity
        (m, _) = A.shape
        # edges_list = [(i, j, {'weight': (A[i, j] + 1e-5)}) for i in range(m) for j in range(m) if
        #               A[i, j] >= threshold and i < j]
        for i in range(m):
            for j in range(m):
                if A[i, j] >= threshold and i < j:
                    edges_list.append((i, j, {'weight': A[i, j]}))
                    node_index.append(i)
                    node_index.append(j)

    G = nx.Graph()
    G.add_edges_from(edges_list)
    node_index = list(set(node_index))
    print("RBFKernelsGraph: number_of_nodes:{0}, number_of_edges:{1}".format(G.number_of_nodes(), G.number_of_edges()))

    # Add  node attributes
    G = set_node_attributes(G, X)
    if G.number_of_nodes() < X.shape[0]:
        print("RBFKernelsGraph Fixing (Isolated Points) ......... ")
        # Fixed Graph
        node_all_index = [nd for nd in range(X.shape[0])]
        node_non_index = list(set(node_all_index) - set(node_index))
        for ndi in node_non_index:
            ptq = X[ndi:ndi + 1, :]
            indices, distances = query_GraphND_cKDTree(G, points=ptq, num_feature=X.shape[1])
            pty = X[indices[0]:indices[0] + 1, :]
            rbf = rbf_kernel(X=ptq, Y=pty, gamma=gamma)
            # add_edge = [(ndi, indices[0], {'weight': 1.0 / (distances[0] + 1e-5)})]
            # add_edge = [(ndi, indices[0], {'weight': rbf[0][0]})]
            w = np.sqrt(np.log(rbf[0][0] + 1e-5) / (-gamma)) / (distances[0] + 1e-5)
            add_edge = [(ndi, indices[0], {'weight': w})]
            G.add_edges_from(add_edge)
            # update the new node attributes
            G = update_node_attributes(G, X, node_new=ndi)
            node_index.append(ndi)
        print("RBFKernelsGraph(Fixed): number_of_nodes:{0}, number_of_edges:{1}".format(G.number_of_nodes(),
                                                                                        G.number_of_edges()))
    # # onnected components in G
    if merage_components:
        connected_components = [list(c) for c in sorted(nx.connected_components(G), key=len, reverse=False)]
        Len_C = len(connected_components)
        print('RBFKernelsGraph Components Numbers: ', Len_C)
        print("RBFKernelsGraph Components Connecting ......... ")
        while Len_C >= 2:
            first_components = connected_components[0]
            # first_sub_graph
            first_sub_graph = G.subgraph(first_components).copy()

            # remove_nodes_from first_sub_graph for G
            G.remove_nodes_from(first_components)

            index_v = [nd for nd, data in first_sub_graph.nodes(data=True)]
            points = X[index_v, :]

            index_us, distances = query_GraphND_cKDTree(G, points=points, num_feature=X.shape[1])

            index_us, distances = list(index_us), list(distances)
            min_index = distances.index(min(distances))  # 返回最小值 index
            v, u, dis = index_v[min_index], index_us[min_index], distances[min_index]

            ptv, ptu = X[v:v + 1, :], X[u:u + 1, :]
            rbf = rbf_kernel(X=ptv, Y=ptu, gamma=gamma)
            w = np.sqrt(np.log(rbf[0][0] + 1e-5) / (-gamma)) / (dis + 1e-5)
            bridge_edge = [(v, u, {'weight': w})]
            # bridge_edge = [(v, u, {'weight': 1.0 / (dis + 1e-5)})] # only distance

            # add bridge edge
            G.add_edges_from(bridge_edge)

            # update the new node attributes
            G = update_node_attributes(G, X, node_new=v)

            # add the subgraph to G
            sub_graph_edges = [edge for edge in first_sub_graph.edges(data=True)]
            G.add_edges_from(sub_graph_edges)
            nodelist = [node for node in G.nodes(data=False)]
            G = set_part_node_attributes(G, nodelist, X)

            connected_components = [list(c) for c in sorted(nx.connected_components(G), key=len, reverse=False)]
            Len_C = len(connected_components)
        print("RBFKernelsGraph(Finally): number_of_nodes:{0}, number_of_edges:{1}".format(G.number_of_nodes(),
                                                                                          G.number_of_edges()))

    if shpfile:
        X_pca = XDecomposition(X, n_components=2)
        if X_pca.shape[1] <= X.shape[1]:
            Edge2Shpfile(G, X_pca, shpfile="edge_shp.shp", data=True, input_crs="EPSG:2436", output_crs=None)
            X2Shpfile(X_pca, shpfile="X_shp.shp", input_crs="EPSG:2436", output_crs=None)
            print("Shpfile is saved")

    return G

# Priori Knowledge 1
def get_pk_MiniBatchKMeans(X, nodes, n_clusters=2, init_size=2, batch_size=500, n_init=10, max_no_improvement=10, verbose=0):
    # # # # # # # # # # # # # # Priori Knowledges # # # # # # # # # # # # # # #
    from sklearn.cluster import MiniBatchKMeans
    from sklearn import metrics
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
        # print('n_clusters', n_clusters)
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
def get_pk_DBSCAN(X, nodes, eps=0.02, min_samples=25):
    # Priori Knowledge 2
    from sklearn.cluster import DBSCAN
    from sklearn import metrics
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels_ = db.labels_
    print("X -> SC score: %0.3f" % metrics.silhouette_score(X, labels_))
    print("X -> CN score: %0.3f" % metrics.calinski_harabasz_score(X, labels_))
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

# get_X_fromSHP
def get_X_fromSHP(filename=r'./park/BJ_park_m.shp', epsg4326_srid=2436, plot=True):
    gdata = gpd.read_file(filename=filename, bbox=None, mask=None, rows=None)#读取磁盘上的矢量文件
    # gdata = gpd.read_file(filename=r'CUsersjeshyDocumentsArcGISJ50.gdb', layer='BOUA')#读取gdb中的矢量数据
    print('Input geodata crs:', gdata.crs)  # 查看数据对应的投影信息(坐标参考系) ---> epsg == 4326  (WGS_84)
    gdata = gdata.to_crs(epsg=epsg4326_srid)# epsg == 2436 (X, Y)
    print('Output geodata crs:', gdata.crs)
    print('Column names', gdata.columns.values)# 列名
    # print(gdata.head())  # 查看前5行数据
    if plot:
        gdata.plot()
        plt.show()#展示

    geo = gdata.geometry.values
    X0, Y0 = geo.x, geo.y
    X = np.zeros(shape=(X0.shape[0], 2), dtype=float)
    for i in range(X0.shape[0]):
        X[i, 0], X[i, 1] = X0[i],  Y0[i]

    print('n_samples:', X0.shape[0])
    output = open('X.pkl', 'wb')
    pickle.dump(X, output)
    output.close()
    print('X.pkl is saved')

    from sklearn.preprocessing import StandardScaler
    X_trans = StandardScaler().fit_transform(X)
    output = open('trans_X.pkl', 'wb')
    pickle.dump(X_trans, output)
    output.close()
    print('trans_X.pkl is saved')

    return X

# get_G_X_formOSM
def get_G_X_formOSM(osmfile=r'./osm/bj_six_ring.osm', simplify=True, gexf_save=True, show_graph=False):
    from networkx import Graph
    from osmnx import graph_from_xml, plot_graph, settings, config
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
    if show_graph:
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

def get_initial_partition(pk_partion):
    initial_partition = {nd: clus for clus in range(len(pk_partion)) for nd in pk_partion[clus]}
    return initial_partition
def findPrioriKnowledges(G, initial_partition=None, edge_data_flg=False):
    import infomap
    infomapX = infomap.Infomap("--two-level")
    # print("Building Infomap network from a NetworkX graph...")

    if edge_data_flg:
        for v, u, w in G.edges(data=edge_data_flg):
            e = (v, u, w['weight'])
            infomapX.addLink(*e)
    else:
        for e in G.edges(data=edge_data_flg):
            infomapX.addLink(*e)

    # print("Find communities with Infomap...")
    infomapX.run(initial_partition=initial_partition, silent=True)
    # print("Found {} modules with codelength: {}".format(infomapX.numTopModules(), infomapX.codelength))
    communities = {}
    for node in infomapX.iterLeafNodes():
        communities[node.physicalId] = node.moduleIndex()
    nx.set_node_attributes(G, values=communities, name='community')
    return infomapX.numTopModules(), communities
# color map from http://colorbrewer2.org/
def drawGraph(G):
    import matplotlib.colors as colors
    # position map
    # pos = nx.spring_layout(G)
    pos = nx.kamada_kawai_layout(G)
    # community ids
    communities = [v for k, v in nx.get_node_attributes(G, 'community').items()]
    numCommunities = max(communities) + 1
    # color map from http://colorbrewer2.org/
    cmapLight = colors.ListedColormap(['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd','#ccebc5','#ffed6f'], 'indexed',
                                      numCommunities)
    cmapDark = colors.ListedColormap(['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928'], 'indexed', numCommunities)
    # Draw edges
    nx.draw_networkx_edges(G, pos)
    # Draw nodes
    nodeCollection = nx.draw_networkx_nodes(G,
                                            pos=pos,
                                            node_color=communities,
                                            cmap=cmapLight)
    # Set node border color to the darker shade
    darkColors = [cmapDark(v) for v in communities]
    nodeCollection.set_edgecolor(darkColors)
    # Draw node labels
    for n in G.nodes():
        plt.annotate(n,
                     xy=pos[n],
                     textcoords='offset points',
                     horizontalalignment='center',
                     verticalalignment='center',
                     xytext=[0, 0],
                     color=cmapDark(communities[n]))
    plt.axis('off')
    # plt.savefig("karate.png")
    plt.show()
# Priori Knowledge 3
def get_PKResult(G, initial_partition=None, gshow=False, edge_data_flg=False):
    # print("Partition by Infomap ..........")
    numModules, pks = findPrioriKnowledges(G, initial_partition=initial_partition, edge_data_flg=edge_data_flg)
    if gshow:
        drawGraph(G)
    list_values = [val for val in pks.values()]
    list_nodes = [node for node in pks.keys()]
    print('pk sortting .......')
    result = []
    for ndpa in range(numModules):
        nodes_part_index = np.argwhere(np.array(list_values) == ndpa).ravel()
        nodes_part = list(np.array(list_nodes)[nodes_part_index])
        nodes_part.sort()
        result.append(nodes_part)
    result.sort()
    # print("pk size:", len(result))
    # print("pk result:", result)

    # output = open('info_pk_partion.pkl', 'wb')
    # pickle.dump(result, output)
    # output.close()

    return result
    
# label_result
def label_result(nsamples, result):
    y_result = np.zeros((nsamples,)) - 1  # -1 为噪声点
    cls = 0
    for p in result:
        y_result[p] = cls
        cls += 1
    return y_result

# MaxDIClustering Mian Class
class MaxDIClustering():
    def __init__(self, affinity='radius',
                 n_neighbors=10,  # getKNNGraph
                 threshold=2.25,  # getKNNGraph, getRBFKernelsGraph
                 radius=2.25,  # getRadiusGraph
                 gamma=0.25,
                 # getRBFKernelsGraph # K(x, y) = exp(-gamma ||x-y||^2) ----> gamma = (2sigma)^-2 ; sigma^2 is the Gaussian kernel of variance.
                 merage_components=True,  # getKNNGraph， getRadiusGraph, getRBFKernelsGraph
                 pk='1',  # -- PK
                 n_clusters=2542, init_size=2542,  # get_pk_MiniBatchKMeans -- PK
                 eps=1000, min_samples=50,  # get_pk_DBSCAN -- PK
                 n_jobs=None):
        self.affinity = affinity
        self.radius = radius
        self.n_neighbors = n_neighbors
        self.threshold = threshold
        self.gamma = gamma  # K(x, y) = exp(-gamma ||x-y||^2) ----> gamma = (2sigma)^-2 ; sigma^2 is the Gaussian kernel of variance.
        self.merage_components = merage_components
        self.n_clusters = n_clusters,
        self.init_size = init_size,
        self.eps = eps
        self.min_samples = min_samples
        self.pk = pk
        self.n_jobs = n_jobs

    def fit(self, X):
        if self.affinity == 'radius':
            self.G = getRadiusGraph(X, radius=self.radius, merage_components=self.merage_components, save=False,
                                    shpfile=False)
        elif self.affinity == 'knn':
            self.G = getKNNGraph(X, n_neighbors=self.n_neighbors, threshold=self.threshold,
                                 merage_components=self.merage_components, save=False, shpfile=False)
        elif self.affinity == 'rbf':  # RBFKernelsGraph # K(x, y) = exp(-gamma ||x-y||^2) ----> gamma = (2sigma)^-2 ; sigma^2 is the Gaussian kernel of variance.
            self.G = getRBFKernelsGraph(X, gamma=self.gamma, threshold=self.threshold,
                                        merage_components=self.merage_components, n_jobs=self.n_jobs, shpfile=False)
        else:
            raise ValueError("Unknown Graph Type %r" % self.affinity)

        self.nodes = np.array([nd for nd in self.G.nodes], dtype=int)

        # Priori Knowledge 1
        if self.pk == '1':
            pk_partion_hat = get_pk_MiniBatchKMeans(X, nodes=self.nodes, n_clusters=self.n_clusters,
                                                init_size=self.init_size)
            # pk_partion_hat = get_PKResult(self.G, initial_partition=get_initial_partition(pk_partion_hat), gshow=False,
            #                               edge_data_flg=True)
            self.pk_DI_ = get_DI_Parallel(self.G, partion=pk_partion_hat, weight='weight', n_jobs=self.n_jobs,
                                          verbose=0)
        # Priori Knowledge 2
        elif self.pk == '2':
            pk_partion_hat = get_pk_DBSCAN(X_trans=X, nodes=self.nodes, eps=self.eps, min_samples=self.min_samples)
            # pk_partion_hat = get_PKResult(self.G, initial_partition=get_initial_partition(pk_partion_hat), gshow=False,
            #                               edge_data_flg=True)
            self.pk_DI_ = get_DI_Parallel(self.G, partion=pk_partion_hat, weight='weight', n_jobs=self.n_jobs,
                                          verbose=0)
        # Priori Knowledge 3
        elif self.pk == '3':
            pk_partion_hat = get_PKResult(self.G, initial_partition=None, gshow=False, edge_data_flg=True)
            self.pk_DI_ = get_DI_Parallel(self.G, partion=pk_partion_hat, weight='weight', n_jobs=self.n_jobs,
                                          verbose=0)
        # None Priori Knowledge
        else:
            pk_partion_hat = None

        self.OSIE_ = get_oneStruInforEntropy(self.G, weight='weight')
        self.OSIE_ = float('%.3f' % self.OSIE_)

        self.results_, self.DIs_ = iter_maxDI(self.G, iter_max=100, pk_partion=pk_partion_hat, weight='weight',
                                             n_jobs=self.n_jobs, verbose=0)
        self.DI_ = self.DIs_[-1]

        self.labels_ = label_result(nsamples=X.shape[0], result=self.results_).astype(int)

        self.DI_rate_ = self.DI_ / self.OSIE_

        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        self.si_score_ = silhouette_score(X, self.labels_, metric='euclidean')
        self.ch_score_ = calinski_harabasz_score(X, self.labels_)

        return self

def test():

    # test
    output = open('park_X.pkl', 'rb')
    X = pickle.load(output)
    output.close()
    output = open('busstop_X.pkl', 'rb')
    X = pickle.load(output)
    output.close()

    # m, n = X.shape

    # show
    print(X.shape)# (42161, 2)
    print(X[[0, 1, 2]])# [[ 451389.26206748 4425262.84838121] .... ]

    # cKDTree
    tree = cKDTree(data=X, leafsize=8, compact_nodes=True, balanced_tree=True)
    X_query = X[[0, 1, 2]]
    dis, index = tree.query(x=X_query, k=1, n_jobs=-1)
    X_query_out = X[index]

    print(dis, index, sep='\n')# [0. 0. 0.] \n [0   1   6931]
    print(X_query)# [[ 451389.26206748 4425262.84838121] ... ]
    print(X_query_out)# [[ 451389.26206748 4425262.84838121] ... ]

    M = 25000# 42161 --> size is too large.
    threshold = 0.998

    params = {}
    params['gama'] = 1
    K = pairwise_kernels(X[:M, :], metric='rbf', filter_params=True, n_jobs=-1, **params)# A kernel matrix K

    # K.tocoo()
    Kcsr = sp.csr_matrix(K)
    K = Kcsr.tocoo()
    print('A kernel matrix K: ', K.shape)
    # K(x, y) = exp(-gamma ||x-y||^2) ----> gamma = (2sigma)^-2 ; sigma^2 is the Gaussian kernel of variance.

    affinity = 0.5 * (K + K.T)
    edges_list, node_index = [], []
    if hasattr(affinity, 'tocoo'):
        A = affinity.tocoo()
        # edges_list = [(i, j, {'weight': (v + 1e-5)}) for i, j, v in zip(A.row, A.col, A.data) if
        #               v >= threshold and i < j]  # coo matrix
        for i, j, v in zip(A.row, A.col, A.data):  # coo matrix
            if v >= threshold and i < j:
                edges_list.append((i, j, {'weight': v}))
                node_index.append(i)
                node_index.append(j)
    # else:
    #     A = affinity
    #     (m, _) = A.shape
    #     # edges_list = [(i, j, {'weight': (A[i, j] + 1e-5)}) for i in range(m) for j in range(m) if
    #     #               A[i, j] >= threshold and i < j]
    #     for i in range(m):
    #         for j in range(m):
    #             if A[i, j] >= threshold and i < j:
    #                 edges_list.append((i, j, {'weight': A[i, j]}))
    #                 node_index.append(i)
    #                 node_index.append(j)

    G = nx.Graph()
    G.add_edges_from(edges_list)
    # node_index = list(set(node_index))
    print("KernelsGraph: number_of_nodes:{0}, number_of_edges:{1}".format(G.number_of_nodes(), G.number_of_edges()))

    # FixingGraph
    G = FixingGraph(G, X, merage_components=True)

    # save
    output = open('busstop_KernelsGraph_part.pkl', 'wb')
    pickle.dump(G, output)
    output.close()

def rbf_Clustering():
    # Load the geodata
    output = open('park_X.pkl', 'rb')# park_X = get_X_fromSHP(filename=r'./park/BJ_park_m.shp', epsg4326_srid=2436, plot=True)
    park_X = pickle.load(output)
    output.close()

    output = open('busstop_X.pkl', 'rb')# busstop_X = get_X_fromSHP(filename=r'./busstop/BeijingBusStops_84.shp', epsg4326_srid=2436, plot=True)
    busstop_X = pickle.load(output)# (42161, 2) # Maybe MemoryError --> A RBF kernel matrix K: (42161, 42161)!
    output.close()

    output = open('./busstop_RBFGraph.pkl', 'rb')
    RBFGraph = pickle.load(output)
    output.close()

    print('------------------------------Extract Partion Outlines----------------------------------')
    # Select Anyone Graph Extracted Method: getRadiusGraph、getKNNGraph、getRBFKernelsGraph
    G = getRadiusGraph(X=park_X, radius=500, merage_components=True, save=False, shpfile=False)
    partion_result = get_PKResult(G, initial_partition=None, gshow=False, edge_data_flg=True)
    outline_gdf = get_partion_result_outline(X=park_X, partion_result=partion_result, input_crs="EPSG:2436",
                                             output_crs=None,
                                             shpfile="partion_outline_shp.shp", save_shp=True, plot=False)
    output = open('Extract_Partion_Outlines_gdf.pkl', 'wb')
    pickle.dump(outline_gdf, output)
    output.close()
    print('------------------------------Extract Partion Outlines End----------------------------------')

    print('------------------------------RBFKernelsGraph----------------------------------')
    print('park_X......')
    X = park_X
    # Clustering by MaxDIClustering
    max_di_clus = MaxDIClustering(affinity='rbf',
                                  threshold=0.998,  # getKNNGraph, getRBFKernelsGraph
                                  gamma=1.0,
                                  # getRBFKernelsGraph # K(x, y) = exp(-gamma ||x-y||^2) ----> gamma = (2sigma)^-2 ; sigma^2 is the Gaussian kernel of variance.
                                  merage_components=True,  # getKNNGraph， getRadiusGraph, getRBFKernelsGraph
                                  pk='3',  # -- PK
                                  n_jobs=4).fit(X)
    output = open('park_X_max_di_clus_rbf.pkl', 'wb')
    pickle.dump(max_di_clus, output)
    output.close()
    print('DI_', max_di_clus.DI_)
    print('OSIE_', max_di_clus.OSIE_)
    print('DI_rate_', max_di_clus.DI_rate_)
    print('si_score_', max_di_clus.si_score_)
    # DI_ 6.775
    # OSIE_ 9.205
    # DI_rate_ 0.736
    # si_score_ 0.299

    print('busstop_X......')
    X = busstop_X  # (42161, 2) # MemoryError: Unable to allocate 13.2 GiB for an array with shape (42161, 42161) and data type float64
    # # X clip by beijing-6ring-road.osm # #
    X_gdf = X2GDF(X, input_crs="EPSG:2436", output_crs=None, plot=None)
    # _, __, gpoly_mask = get_poly_from_osm_circleline(circleline_osmfile=r"../osm/beijing_car_seg_6ring road.osm", toepsg=2436)  # (23946, 2) # MemoryError too
    _, __, gpoly_mask = get_poly_from_osm_circleline(circleline_osmfile=r"../osm/beijing_5ring.osm", toepsg=2436)
    gdata = gpd.clip(gdf=X_gdf, mask=gpoly_mask, keep_geom_type=True)
    X = GDF2X(gdata, to_epsg=2436, plot=False)
    # Clustering by MaxDIClustering
    max_di_clus = MaxDIClustering(affinity='rbf',
                                  threshold=0.998,  # getKNNGraph, getRBFKernelsGraph
                                  gamma=1.0,
                                  # getRBFKernelsGraph # K(x, y) = exp(-gamma ||x-y||^2) ----> gamma = sigma^-2 ; sigma^2 is the Gaussian kernel of variance.
                                  merage_components=True,  # getKNNGraph， getRadiusGraph, getRBFKernelsGraph
                                  pk='3',  # -- PK
                                  n_jobs=4).fit(X)
    output = open('busstop_X_max_di_clus_rbf.pkl', 'wb')
    pickle.dump(max_di_clus, output)
    output.close()
    print('DI_', max_di_clus.DI_)
    print('OSIE_', max_di_clus.OSIE_)
    print('DI_rate_', max_di_clus.DI_rate_)
    print('si_score_', max_di_clus.si_score_)
    # A kernel matrix K:  (13318, 13318)
    # RBFKernelsGraph: number_of_nodes:8278, number_of_edges:15797
    # RBFKernelsGraph Fixing (Isolated Points) .........
    # RBFKernelsGraph(Fixed): number_of_nodes:13318, number_of_edges:20837
    # RBFKernelsGraph Components Numbers:  2271
    # RBFKernelsGraph Components Connecting .........
    # RBFKernelsGraph(Finally): number_of_nodes:13318, number_of_edges:23107
    # DI_ 7.862
    # OSIE_ 9.827
    # DI_rate_ 0.800
    # si_score_ 0.694

    G = RBFGraph  # 提前使用更高配置计算机计算得到的 RBF-Graph
    X = busstop_X  # (42161, 2)
    # FixingGraph
    G = FixingGraph(G, X, merage_components=True)

    pk_partion_hat = get_PKResult(G, initial_partition=None, gshow=False, edge_data_flg=True)
    OSIE_ = float('%.3f' % get_oneStruInforEntropy(G, weight='weight'))

    # pk_DI_ = get_DI_Parallel(G, partion=pk_partion_hat, weight='weight', n_jobs=-1, verbose=0)
    results_, DI_ = iter_maxDI(G, iter_max=100, pk_partion=pk_partion_hat, weight='weight', n_jobs=-1, verbose=0)

    DI_ = DI_[-1]
    labels_ = label_result(nsamples=X.shape[0], result=results_).astype(int)
    DI_rate_ = DI_ / OSIE_
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    si_score_ = silhouette_score(X, labels_, metric='euclidean')
    ch_score_ = calinski_harabasz_score(X, labels_)
    print('DI_', DI_)
    print('OSIE_', OSIE_)
    print('DI_rate_', DI_rate_)
    print('si_score_', si_score_)
    # DI_ 8.751
    # OSIE_ 10.52
    # DI_rate_ 0.832
    # si_score_ 0.587
    print('------------------------------RBFKernelsGraph - end----------------------------------')

def main():
    # 1. test
    # test()

    # 2. rbf_Clustering
    rbf_Clustering()

    # Load the geodata
    output = open('park_X.pkl', 'rb')# park_X = get_X_fromSHP(filename=r'./park/BJ_park_m.shp', epsg4326_srid=2436, plot=True)
    park_X = pickle.load(output)
    output.close()

    output = open('busstop_X.pkl', 'rb')# busstop_X = get_X_fromSHP(filename=r'./busstop/BeijingBusStops_84.shp', epsg4326_srid=2436, plot=True)
    busstop_X = pickle.load(output)# (42161, 2) # Maybe MemoryError --> A RBF kernel matrix K: (42161, 42161)!
    output.close()

    print('------------------------------Extract Partion Outlines----------------------------------')
    # Select Anyone Graph Extracted Method: getRadiusGraph、getKNNGraph、getRBFKernelsGraph
    G = getRadiusGraph(X=park_X, radius=500, merage_components=True, save=False, shpfile=False)
    partion_result = get_PKResult(G, initial_partition=None, gshow=False, edge_data_flg=True)
    outline_gdf = get_partion_result_outline(X=park_X, partion_result=partion_result, input_crs="EPSG:2436",
                                             output_crs=None,
                                             shpfile="partion_outline_shp.shp", save_shp=True, plot=False)
    output = open('Extract_Partion_Outlines_gdf.pkl', 'wb')
    pickle.dump(outline_gdf, output)
    output.close()
    print('------------------------------Extract Partion Outlines End----------------------------------')

    # Clustering for demonstration
    X = park_X
    # Clustering by MaxDIClustering
    max_di_clus = MaxDIClustering(affinity='rbf',
                                  threshold=0.998,  # other: getKNNGraph, getRBFKernelsGraph
                                  gamma=1.0,
                                  # for getRBFKernelsGraph # K(x, y) = exp(-gamma ||x-y||^2) ----> gamma = (2sigma)^-2 ; sigma^2 is the Gaussian kernel of variance.
                                  merage_components=True,  # other: getKNNGraph， getRadiusGraph, getRBFKernelsGraph
                                  pk='3',  # -- PK
                                  n_jobs=4).fit(X)
    print('DI_', max_di_clus.DI_)
    print('OSIE_', max_di_clus.OSIE_)
    print('DI_rate_', max_di_clus.DI_rate_)
    print('si_score_', max_di_clus.si_score_)

if __name__ == '__main__':
    main()
