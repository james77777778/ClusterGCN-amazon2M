from pathlib import Path
import json
import itertools
import ijson
import metis
import numpy as np
import networkx as nx


dataset = 'reddit'
cluster_parts = 5

with open(Path('datasets', dataset, dataset+'-id_map.json'), 'r') as f:
    json_id_map = json.load(f)
    # check node id type
    for k in json_id_map.keys():
        try:
            int(k)

            def conversion(n):
                return int(n)
            break
        except ValueError:
            def conversion(n):
                return n
            break
    id_map = {conversion(k): int(v) for k, v in json_id_map.items()}
    ids = [v for k, v in json_id_map.items()]
    del json_id_map

with open(Path('datasets', dataset, dataset+'-class_map.json'), 'r') as f:
    json_class_map = json.load(f)
    # check class type
    for k, v in json_class_map.items():
        if isinstance(v, list):
            def lab_conversion(n):
                return n
            break
        else:
            def lab_conversion(n):
                return int(n)
            break
    class_map = {conversion(k): lab_conversion(v) for k, v in json_class_map.items()}
    del json_class_map


# parse nodes
G = nx.Graph()
with open(Path('datasets', dataset, dataset+'-G.json'), 'r') as f:
    json_nodes = ijson.items(f, 'nodes.item')
    for node in json_nodes:
        node_id = id_map[node['id']]
        G.add_node(node_id, attr_dict={'val': node['val'], 'test': node['test']})
num_data = len(id_map)
val_nodes = np.array([n for n in G.nodes() if G.node[n]['val']], dtype=np.int32)
test_nodes = np.array([n for n in G.nodes() if G.node[n]['test']], dtype=np.int32)
is_train = np.ones((num_data), dtype=np.bool)
is_train[val_nodes] = False
is_train[test_nodes] = False
train_nodes = np.array([n for n in range(num_data) if is_train[n]], dtype=np.int32)
print('nodes finished.')

# parse edges
with open(Path('datasets', dataset, dataset+'-G.json'), 'r') as f:
    json_edges = ijson.items(f, 'links.item')
    for i, edge in enumerate(json_edges):
        G.add_edge(edge['source'], edge['target'])
print('edges finished.')
# all edges
edges = list(G.edges())
# train edges
train_edges = [(e[0], e[1]) for e in edges if is_train[e[0]] and is_train[e[1]]]

print(nx.info(G))
train_G = nx.convert_node_labels_to_integers(
    G.subgraph(train_nodes), label_attribute='ori_label')
print(nx.info(train_G))
print('start partition')
METIS_DBG_ALL = sum(2**i for i in list(range(9))+[11])
(edgecuts, parts) = metis.part_graph(train_G.adjacency_list(), cluster_parts, dbglvl=METIS_DBG_ALL)
parts = np.array(parts, dtype=np.int32)
landmarks_in_train_G = []
for i in range(cluster_parts):
    parts_idx = np.where(parts == i)[0]
    parts_graph = train_G.subgraph(parts_idx)
    # find top-1% nodes
    landmarks = []
    for node, degree in sorted(parts_graph.degree().items(), key=lambda item: item[1], reverse=True):
        landmarks.append(node)
        if len(landmarks) > parts_graph.number_of_nodes() * 0.01:
            landmarks_in_train_G.extend(landmarks)
            break
    # save partial graph with original label
    mapping = {n: parts_graph.node[n]['ori_label'] for n in parts_graph.nodes()}
    parts_graph = nx.relabel_nodes(parts_graph, mapping, copy=True)
    for (n, data) in parts_graph.nodes(data=True):
        data.pop('ori_label', None)
    with open(Path('clustering', dataset, '{}.gpickle'.format(i)), 'wb') as f:
        nx.write_gpickle(parts_graph, f)

combinations = list(itertools.combinations(landmarks_in_train_G, 2))
combinations = set(tuple(sorted(item)) for item in combinations)

landmark_graph = nx.Graph()
for u, v, p in nx.adamic_adar_index(train_G, combinations):
    ori_u = train_G.node[u]['ori_label']
    ori_v = train_G.node[v]['ori_label']
    edge_w = 2./(1 + np.exp(-p/10)) - 1.
    landmark_graph.add_edge(ori_u, ori_v, attr_dict={'weight': edge_w})
    landmark_graph.add_edge(ori_v, ori_u, attr_dict={'weight': edge_w})

with open(Path('clustering', dataset, 'landmark.gpickle'), 'wb') as f:
    nx.write_gpickle(landmark_graph, f)
