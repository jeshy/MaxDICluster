import pickle
import numpy as np
import networkx as nx
import osmnx as ox
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiLineString
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from scipy.spatial import cKDTree

def query_cKDTree(X, point_query):
    # X --> m x n; point_query  --> k x n
    # build a k-d tree for euclidean nearest node search
    tree = cKDTree(data=X, compact_nodes=True, balanced_tree=True)
    # X_q, Y_q = X[0:10, 0]+50, X[00:10, 1]+50
    # query the tree for nearest node to each point
    # point_query = np.array([X_q, Y_q]).T
    dist, idx = tree.query(point_query, k=1)
    return dist, idx

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

def X2Shpfile(X, shpfile = "X_shp.shp", input_crs = "EPSG:2436", output_crs = "EPSG:4326"):
    # input_crs = "EPSG:2436"# input_crs = "EPSG:4326"
    # output_crs = "EPSG:4326"# output_crs = "EPSG:2436"
    # shpfile = "X_shp.shp"
    coords, IDs = [], []
    for i in range(X.shape[0]):
        coords.append(Point(X[i, 0], X[i, 1]))# (X, Y)
        IDs.append("%d" % i)
    data = {'ID': IDs, 'geometry': coords}
    gdf = gpd.GeoDataFrame(data, crs=input_crs)
    # print(gdf.crs)  # ?????????????????????????????????(???????????????)
    if output_crs is not None:
        gdf.to_crs(crs=output_crs)# (Lon, Lat)
    gdf.to_file(shpfile)

def Edge2Shpfile(G, X, shpfile = "edge_shp.shp", data=True, input_crs = "EPSG:2436", output_crs = "EPSG:4326"):
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
    # print(gdf.crs)  # ?????????????????????????????????(???????????????)
    if output_crs is not None:
        gdf.to_crs(crs=output_crs)
    gdf.to_file(shpfile)

def getKNNGraph(X, n_neighbors=10, threshold=2.25, merage_components=False, save=True, shpfile=False):
    # threshold = threshold  ########### setup parameter ###########
    # X Decomposition
    X_pca = XDecomposition(X, n_components=2)

    # X_trans = StandardScaler().fit_transform(X)

    if save:
        output = open('X.pkl', 'wb')
        pickle.dump(X, output)
        output.close()
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
            min_index = distances.index(min(distances))# ??????????????? index
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
        if X_pca.shape[1] <= X.shape[1]:
            Edge2Shpfile(G, X_pca, shpfile="edge_shp.shp", data=True, input_crs="EPSG:2436", output_crs=None)
            X2Shpfile(X_pca, shpfile="X_shp.shp", input_crs="EPSG:2436", output_crs=None)
            print("Shpfile is saved")

    return G

def getRadiusGraph(X, radius=2.25, merage_components=False, save=True, shpfile=False):
    # threshold = threshold  ########### setup parameter ###########
    # X Decomposition
    X_pca = XDecomposition(X, n_components=2)

    # X_trans = StandardScaler().fit_transform(X)

    if save:
        output = open('X.pkl', 'wb')
        pickle.dump(X, output)
        output.close()
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
            min_index = distances.index(min(distances))# ??????????????? index
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
        if X_pca.shape[1] <= X.shape[1]:
            Edge2Shpfile(G, X_pca, shpfile="edge_shp.shp", data=True, input_crs="EPSG:2436", output_crs=None)
            X2Shpfile(X_pca, shpfile="X_shp.shp", input_crs="EPSG:2436", output_crs=None)
            print("Shpfile is saved")

    return G

def get_X_fromSHP(filename=r'./park/BJ_park_m.shp', epsg4326_2436=2436, plot=True):
    gdata = gpd.read_file(filename=filename, bbox=None, mask=None, rows=None)#??????????????????????????????
    # gdata = gpd.read_file(filename=r'CUsersjeshyDocumentsArcGISJ50.gdb', layer='BOUA')#??????gdb??????????????????
    print(gdata.crs)  # ?????????????????????????????????(???????????????) ---> epsg == 4326  (WGS_84)
    gdata = gdata.to_crs(epsg=epsg4326_2436)# epsg == 2436 (X, Y)
    print(gdata.crs)
    print(gdata.columns.values)# ??????
    # print(gdata.head())  # ?????????5?????????
    if plot:
        gdata.plot()
        plt.show()#??????

    geo = gdata.geometry.values
    X0, Y0 = geo.x, geo.y
    X = np.zeros(shape=(X0.shape[0], 2), dtype=float)
    for i in range(X0.shape[0]):
        X[i, 0], X[i, 1] = X0[i],  Y0[i]

    print('n_samples:', X0.shape[0])
    output = open('park_X.pkl', 'wb')
    pickle.dump(X, output)
    output.close()

    X_trans = StandardScaler().fit_transform(X)
    output = open('trans_park_X.pkl', 'wb')
    pickle.dump(X_trans, output)
    output.close()

    return X, X_trans

########################## Test #################################
def main():
    # get lon, lat in shapefile
    X, X_trans = get_X_fromSHP(filename=r'./park/BJ_park_m.shp', epsg4326_2436=2436, plot=True)
    # get Graph
    G = getKNNGraph(X, n_neighbors=100, threshold=500, merage_components=True, save=True, shpfile=True)
    G = getRadiusGraph(X, radius=500, merage_components=True, save=True, shpfile=True)

if __name__ == '__main__':
    main()