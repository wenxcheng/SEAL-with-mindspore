from __future__ import print_function #在python2.x版本中按照3.x版本的方式使用print函数
import numpy as np
import random
from tqdm import tqdm #使用进度条展示训练过程
import os, sys, pdb, math, time
import networkx as nx
import argparse
import scipy.io as sio
import scipy.sparse as ssp
from sklearn import metrics
from gensim.models import Word2Vec
import warnings
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning) #忽略警告及类型
cur_dir = os.path.dirname(os.path.realpath(__file__)) #获取文件所在的绝对路径
sys.path.append('%s/../pytorch_DGCNN' % cur_dir)
sys.path.append('%s/software/node2vec/src' % cur_dir)
from util import GNNGraph
import node2vec
import multiprocessing as mp
from itertools import islice

def sample_neg(net, test_ratio=0.1, train_pos=None, test_pos=None, max_train_num=None, all_unknown_as_negative=False):
    #获取上三角矩阵
    net_triu = ssp.triu(net, k=1)
    #采样正连接
    row, col, _ = ssp.find(net_triu)
    #未指定则采样正连接
    if train_pos is None and test_pos is None:
        perm = random.sample(range(len(row)), len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio))) #向上舍入再取整
        train_pos = (row[:split], col[:split]) #取[0:split]项
        test_pos = (row[split:], col[split:])
    #若指定了max_train_num，随机采样训练连接
    if max_train_num is not None and train_pos is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])
    #采样负连接
    train_num = len(train_pos[0]) if train_pos else 0
    test_num = len(test_pos[0]) if test_pos else 0
    neg = ([], [])
    n = net.shape[0]
    print('sampling negtive links for train and test')
    if not all_unknown_as_negative:
        #采样一部分未知连接作为train_negs和test_negs(无重叠)
        while len(neg[0]) < train_num + test_num:
            i,j = random.randint(0, n-1), random.randint(0, n-1)
            if i < j and net[i, j] == 0:
                neg[0].append(i)
                neg[1].append(j)
            else:
                continue
        train_neg = (neg[0][:train_num], neg[1][:train_num])
        test_neg = (neg[0][train_num:], neg[1][train_num:])
    else:
        #将所有未知连接视为test_negs, 从中提取一部分作为train_negs
        while len(neg[0]) < train_num:
            i, j = random.randint(0, n-1), random.randint(0, n-1)
            if i < j and net[i, j] == 0:
                neg[0].append(i)
                neg[1].append(j)
            else:
                continue
        train_neg = (neg[0], neg[1])
        test_neg_i, test_neg_j, _ = ssp.find(ssp.triu(net==0, k=1))
        test_neg = (test_neg_i.tolist(), test_neg_j.tolist())
    return train_pos, train_neg, test_pos, test_neg


def links2subgraphs(A, train_pos, train_neg, test_pos, test_neg, h=1,
                   max_nodes_per_hop=None, node_information=None, no_parallel=False):
    #从{1,2}中自动选择h
    if h == 'auto':
        #split train into val_train and val_test
        _, _, val_test_pos, val_test_neg = sample_neg(A, 0.1)
        val_A = A.copy()
        val_A[val_test_pos[0], val_test_pos[1]] = 0
        val_A[val_test_pos[1], val_test_pos[0]] = 0

    #提取封闭子图
    max_n_label = {'value':0}
    def helper(A, links, g_label):
        g_list = []
        if no_parallel:
            for i,j in tqdm(zip(links[0], links[1])):
                g, n_labels, n_features = subgraph_extraction_labeling(
                    (i, j), A, h, max_nodes_per_hop, node_information
                )
                max_n_label['value'] = max(max(n_labels), max_n_label['value'])
                g_list.append(GNNGraph(g, g_label, n_labels, n_features))
            return g_list
        else:
            #并行提取
            start = time.time()
            pool = mp.Pool(mp.cpu_count())
            results = pool.map_async(
                parallel_worker,
                [((i, j), A, h, max_nodes_per_hop, node_information) for i,j in zip(links[0], links[1])]
            )
            remaining = results._number_left
            pbar = tqdm(total=remaining)
            while True:
                pbar.update(remaining - results._number_left)
                if results.ready(): break
                remaining = results._number_left
                time.sleep(1)
            results = results.get()
            pool.close()
            pbar.close()
            g_list = [GNNGraph(g, g_label, n_labels, _ in results) for g, n_labels, n_features in results]
            max_n_label['value'] = max(
                max([max(n_labels) for _, n_labels, _ in results]), max_n_label['value']
            )
            end = time.time()
            print("Time eplased for subgraph extraction: {}s".format(end-start))
            return g_list
    print('Enclosing subgraph extraction begins..')
    train_graphs, test_graphs = None, None
    if train_pos and train_neg:
        train_graphs = helper(A, train_pos, 1) + helper(A, train_neg, 0)
    if test_pos and test_neg:
        test_graphs = helper(A, test_pos, 1) + helper(A, test_neg, 0)
    elif test_pos:
        test_graphs = helper(A, test_pos, 1)
    return  train_graphs, test_graphs, max_n_label['value']


def parallel_worker(x):
    return subgraph_extraction_labeling(*x)


def subgraph_extraction_labeling(ind, A, h=1, max_node_per_hop=None, node_information=None):
    #提取连接ind周围的h跳封闭子图
    dist = 0
    nodes = set([ind[0], ind[1]])
    visited = set([ind[0], ind[1]])
    fringe = set([ind[0], ind[1]])
    nodes_dist = [0, 0]
    for dist in range(1, h+1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if max_node_per_hop is not None:
            if max_node_per_hop < len(fringe):
                fringe = random.sample(fringe, max_node_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
        nodes_dist += [dist] * len(fringe)
    #将目标节点移到顶部
    nodes.remove(ind[0])
    nodes.remove(ind[1])
    nodes = [ind[0], ind[1]] + list(nodes)
    subgraph = A[nodes, :] [:, nodes]
    #节点标记
    labels = node_label(subgraph)
    #获取节点特征
    features = None
    if node_information is not None:
        features = node_information[nodes]
    #建立nx图
    g = nx.from_scipy_sparse_matrix(subgraph)

    #移除目标节点之间的连接
    if g.has_edge(0, 1):
        g.remove_edge(0, 1)
    return g, labels.tolist(), features


def neighbors(fringe, A):
    #在A中fringe边框中找到所有节点的一跳邻居
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res


def node_label(subgraph):
    #DRNL的应用
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0]+list(range(2,K)), :][:, [0]+list(range(2,K))]
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
    dist_to_0 = dist_to_0[1:, 0]
    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
    dist_to_1 = dist_to_0[1:, 0]
    d = (dist_to_0 + dist_to_1).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.asarray[1,1]), labels)
    labels[np.isinf(labels)] = 0
    labels[labels>1e6] = 0 #set inf labels to 0
    labels[labels<-1e6] = 0 #set -inf to 0
    return labels


def generate_node2vec_embeddings(A, emd_size=128, negative_injection=False, train_neg=None):
    if negative_injection:
        row, col = train_neg
        A = A.copy()
        A[row, col] = 1 #inject negative train
        A[col, row] = 1 #inject negative train
    nx_G = nx.from_scipy_sparse_matrix(A)
    G = node2vec.Graph(nx_G, is_directed=False, p=1, q=1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks=10, walk_length=80)
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=emd_size, window=10, min_count=0, sg=1, workers=8, iter=1)
    wv = model.wv
    embeddings = np.zeros([A.shape[0], emd_size], dtype='float32')
    sum_embeddings = 0
    empty_list = []
    for i in range(A.shape[0]):
        if str(i) in wv:
            embeddings[i] = wv.word_vec(str(i))
            sum_embeddings += embeddings[i]
        else:
            empty_list.append(i)
    mean_embedding = sum_embeddings / (A.shape[0] - len(empty_list))
    embeddings[empty_list] = mean_embedding
    return embeddings


def CalAUC(sim, test_pos, test_neg):
    pos_scores = np.asarray(sim[test_pos[0], test_pos[1]]).squeeze() #转换为矩阵后改变维度
    neg_scores = np.asarray(sim[test_neg[0], test_neg[1]]).squeeze()
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])#矩阵合并
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

