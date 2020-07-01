import json
import metis
import numpy as np
import sklearn.metrics
import scipy.sparse as sp
import networkx as nx
from networkx.readwrite import json_graph
import sklearn.preprocessing
import tensorflow as tf


np.random.seed(0)


def load_data(dataset, path='./datasets', normalize=True):
    # Load data files
    G = json_graph.node_link_graph(json.load(tf.io.gfile.GFile('{}/{}/{}-G.json'.format(path, dataset, dataset))))
    id_map = json.load(tf.io.gfile.GFile('{}/{}/{}-id_map.json'.format(path, dataset, dataset)))
    if isinstance(list(G.nodes())[0], int):
        def conversion(n):
            return int(n)
    else:
        def conversion(n):
            return n
    id_map = {conversion(k): int(v) for k, v in id_map.items()}
    class_map = json.load(tf.io.gfile.GFile('{}/{}/{}-class_map.json'.format(path, dataset, dataset)))
    if isinstance(list(class_map.values())[0], list):
        def lab_conversion(n):
            return n
    else:
        def lab_conversion(n):
            return int(n)
    class_map = {conversion(k): lab_conversion(v) for k, v in class_map.items()}
    # Remove broken nodes
    broken_count = 0
    to_remove = []
    for node in G.nodes():
        if node not in id_map:
            to_remove.append(node)
            broken_count += 1
    for node in to_remove:
        G.remove_node(node)
    print(nx.info(G))
    # Generate edge list
    edges = []
    for edge in G.edges():
        if edge[0] in id_map and edge[1] in id_map:
            edges.append((id_map[edge[0]], id_map[edge[1]]))
    # Total Number of Nodes in the Graph
    num_data = len(id_map)

    # Seperate Train, Val, and Test nodes
    val_nodes = np.array([id_map[n] for n in G.nodes() if G.node[n]['val']], dtype=np.int32)
    test_nodes = np.array([id_map[n] for n in G.nodes() if G.node[n]['test']], dtype=np.int32)
    is_train = np.ones((num_data), dtype=np.bool)
    is_train[val_nodes] = False
    is_train[test_nodes] = False
    train_nodes = np.array([n for n in range(num_data) if is_train[n]], dtype=np.int32)
    # Train Edges
    train_edges = [(e[0], e[1]) for e in edges if is_train[e[0]] and is_train[e[1]]]
    train_edges = np.array(train_edges, dtype=np.int32)
    # All Edges in the Graph
    edges = np.array(edges, dtype=np.int32)

    # Generate Labels
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
        labels = np.zeros((num_data, num_classes), dtype=np.float32)
        for k in class_map.keys():
            labels[id_map[k], :] = np.array(class_map[k])
    else:
        num_classes = len(set(class_map.values()))
        labels = np.zeros((num_data, num_classes), dtype=np.float32)
        for k in class_map.keys():
            labels[id_map[k], class_map[k]] = 1

    # train_ids = np.array([id_map[n] for n in G.nodes() if not G.nodes[n]['val'] and not G.nodes[n]['test']])

    # clear mem
    feats = np.load(tf.io.gfile.GFile('{}/{}/{}-feats.npy'.format(path, dataset, dataset), 'rb')).astype(np.float32)
    if normalize:
        train_ids = np.array([
            id_map[n]
            for n in G.nodes()
            if not G.node[n]['val'] and not G.node[n]['test']
        ])
        print("Before feature normalization:", feats[0][0])
        train_feats = feats[train_ids]
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
        print("After feature normalization:", feats[0][0])

    def _construct_adj(e):
        adj = sp.csr_matrix((np.ones((e.shape[0]), dtype=np.float32), (e[:, 0], e[:, 1])), shape=(num_data, num_data))
        adj += adj.transpose()
        return adj

    train_adj = _construct_adj(train_edges)
    full_adj = _construct_adj(edges)

    train_feats = feats[train_nodes]
    test_feats = feats

    visible_nodes = train_nodes

    # Generate Labels
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_nodes, :] = labels[train_nodes, :]
    y_val[val_nodes, :] = labels[val_nodes, :]
    y_test[test_nodes, :] = labels[test_nodes, :]

    # Generate Masks for Validtion & Testing Data
    train_mask = sample_mask(train_nodes, labels.shape[0])
    val_mask = sample_mask(val_nodes, labels.shape[0])
    test_mask = sample_mask(test_nodes, labels.shape[0])

    # Precalc
    train_feats = train_adj.dot(feats)
    train_feats = np.hstack((train_feats, feats))
    test_feats = full_adj.dot(feats)
    test_feats = np.hstack((test_feats, feats))

    return (train_adj, full_adj, train_feats, test_feats, y_train, y_val, y_test,
            train_mask, val_mask, test_mask, train_nodes, val_nodes, test_nodes,
            num_data, visible_nodes)


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def partition_graph(adj, idx_nodes, num_clusters):
    num_nodes = len(idx_nodes)
    num_all_nodes = adj.shape[0]

    neighbor_intervals = []
    neighbors = []
    edge_cnt = 0
    neighbor_intervals.append(0)
    train_adj_lil = adj[idx_nodes, :][:, idx_nodes].tolil()
    train_ord_map = dict()
    train_adj_lists = [[] for _ in range(num_nodes)]

    for i in range(num_nodes):
        rows = train_adj_lil[i].rows[0]
        # self-edge needs to be removed for valid format of METIS
        if i in rows:
            rows.remove(i)
        train_adj_lists[i] = rows
        neighbors += rows
        edge_cnt += len(rows)
        neighbor_intervals.append(edge_cnt)
        train_ord_map[idx_nodes[i]] = i
    if num_clusters > 1:
        _, groups = metis.part_graph(train_adj_lists, num_clusters, seed=1)
    else:
        groups = [0] * num_nodes

    part_row = []
    part_col = []
    part_data = []
    parts = [[] for _ in range(num_clusters)]

    for nd_idx in range(num_nodes):
        gp_idx = groups[nd_idx]
        nd_orig_idx = idx_nodes[nd_idx]
        parts[gp_idx].append(nd_orig_idx)

        for nb_orig_idx in adj[nd_orig_idx].indices:
            nb_idx = train_ord_map[nb_orig_idx]
            if groups[nb_idx] == gp_idx:
                part_data.append(1)
                part_row.append(nd_orig_idx)
                part_col.append(nb_orig_idx)
    part_data.append(0)
    part_row.append(num_all_nodes - 1)
    part_col.append(num_all_nodes - 1)
    part_adj = sp.coo_matrix((part_data, (part_row, part_col))).tocsr()

    return part_adj, parts


def preprocess(adj, features, y_train, train_mask, visible_data, num_clusters, diag_lambda=1):
    # graph partitioning
    part_adj, parts = partition_graph(adj, visible_data, num_clusters)
    part_adj = normalize_adj_diag_enhance(part_adj, diag_lambda=diag_lambda)
    parts = [np.array(pt) for pt in parts]

    features_batches = []
    support_batches = []
    y_train_batches = []
    train_mask_batches = []
    total_nnz = 0
    for pt in parts:
        features_batches.append(features[pt, :])
        now_part = part_adj[pt, :][:, pt]
        total_nnz += now_part.count_nonzero()
        support_batches.append(sparse_to_tuple(now_part))
        y_train_batches.append(y_train[pt, :])

        train_pt = []
        for newidx, idx in enumerate(pt):
            if train_mask[idx]:
                train_pt.append(newidx)
        train_mask_batches.append(sample_mask(train_pt, len(pt)))
    return parts, features_batches, support_batches, y_train_batches, train_mask_batches


def preprocess_multicluster(adj, parts, features, y_train, train_mask, num_clusters, block_size, diag_lambda=1):
    """ Generate batches for multiple clusters."""
    features_batches = []
    support_batches = []
    y_train_batches = []
    train_mask_batches = []
    total_nnz = 0
    np.random.shuffle(parts)

    for _, st in enumerate(range(0, num_clusters, block_size)):
        pt = parts[st]
        for pt_idx in range(st + 1, min(st + block_size, num_clusters)):
            pt = np.concatenate((pt, parts[pt_idx]), axis=0)
        features_batches.append(features[pt, :])
        y_train_batches.append(y_train[pt, :])
        support_now = adj[pt, :][:, pt]
        support_batches.append(sparse_to_tuple(normalize_adj_diag_enhance(support_now, diag_lambda=diag_lambda)))
        total_nnz += support_now.count_nonzero()

        train_pt = []
        for newidx, idx in enumerate(pt):
            if train_mask[idx]:
                train_pt.append(newidx)
        train_mask_batches.append(sample_mask(train_pt, len(pt)))

    return features_batches, support_batches, y_train_batches, train_mask_batches


def normalize_adj_diag_enhance(adj, diag_lambda=1):
    """ A'=(D+I)^{-1}(A+I), A'=A'+lambda*diag(A') """
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = 1.0 / (rowsum + 1e-20)
    d_mat_inv = sp.diags(d_inv, 0)
    adj = d_mat_inv.dot(adj)
    adj = adj + diag_lambda * sp.diags(adj.diagonal(), 0)
    return adj


def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def sample_mask(idx, mat):
    mask = np.zeros(mat)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def calc_f1(y_pred, y_true, multilabel):
    if multilabel:
        y_pred[y_pred > 0] = 1
        y_pred[y_pred <= 0] = 0
    return (sklearn.metrics.f1_score(y_true, y_pred, average='micro'),
            sklearn.metrics.f1_score(y_true, y_pred, average='macro'))
