import time
import pickle
import numpy as np
import networkx as nx
from networkx import Graph
# import geopandas as gpd
# from shapely.geometry import Point
import xml.etree.cElementTree as et
from osmnx import graph_from_xml, plot_graph, settings, config
from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN
from utils_SI import iter_maxDI, label_result
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

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

root = et.parse(r"./osm/bj_six_ring.osm").getroot()
nodes_dict = {}
for node in root.findall('node'):
    nodes_dict[int(node.attrib['id'])] = (float(node.attrib['lon']), float(node.attrib['lat']))

M = graph_from_xml('./osm/bj_six_ring.osm', simplify=True)
# print('G: number_of_nodes:', M.number_of_nodes(), 'number_of_edges:', M.number_of_edges())

# # plot_graph
# plot_graph(M,
#             figsize=(16, 16),
#             bgcolor=None,
#             node_color="#999999", node_size=15, node_alpha=None, node_edgecolor="none",
#             edge_color="#999999", edge_linewidth=1, edge_alpha=None,
#             dpi=600,
#             save=True, filepath='./data/osm/bj_six_ring.png')

# Convert M to Graph G
G = nx.Graph()
G.add_edges_from(Graph(M).edges())
print("G: number_of_nodes:{0}, number_of_edges:{1}".format(G.number_of_nodes(), G.number_of_edges()))

# write_gexf
nx.write_gexf(G, 'bj_six_ring.gexf')

# relabel_nodes
mapping, index, X = {}, 0, []
for node in G.nodes:
    mapping[node] = index
    lon, lat = nodes_dict[node][0], nodes_dict[node][1]
    X.append(np.array([lon, lat], dtype=float))
    index += 1
X = np.array(X, dtype=float)
G = nx.relabel_nodes(G, mapping)
nodes = np.array([nd for nd in G.nodes], dtype=int)

X_trans = StandardScaler().fit_transform(X)

output = open('X.pkl', 'wb')
pickle.dump(X, output)
output.close()

output = open('X_trans.pkl', 'wb')
pickle.dump(X_trans, output)
output.close()

output = open('G.pkl', 'wb')
pickle.dump(G, output)
output.close()