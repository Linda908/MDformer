import os

import dgl
import numpy as np
import torch
from sklearn.model_selection import KFold

from utils import *


# ***************************载入图的相似性特征开始**************************
def similarity_feature_process(args):
    similarity_feature = {}
    path = args.path
    device = args.device

    n_rna = np.loadtxt(os.path.join(path, 'm_gs.csv'), delimiter=',', dtype=float).shape[0]
    n_dis = np.loadtxt(os.path.join(path, 'd_gs.csv'), delimiter=',', dtype=float).shape[0]

    "miRNA sequence sim"
    rna_seq_sim = np.loadtxt(os.path.join(path, 'm_ss.csv'), delimiter=',', dtype=float)
    rna_seq_sim = torch.tensor(rna_seq_sim, device=device).to(torch.float32)
    rna_seq_edge_index = get_edge_index(rna_seq_sim, device)
    g_mm_s = dgl.graph((rna_seq_edge_index[0], rna_seq_edge_index[1]))
    similarity_feature['mm_s'] = {'Data_M': rna_seq_sim, 'edges': rna_seq_edge_index, 'g': g_mm_s}
    "miRNA Gaussian sim"
    rna_Gaus_sim = np.loadtxt(os.path.join(path, 'm_gs.csv'), delimiter=',', dtype=float)
    rna_Gaus_sim = torch.tensor(rna_Gaus_sim, device=device).to(torch.float32)
    rna_Gaus_edge_index = get_edge_index(rna_Gaus_sim, device)
    g_mm_g = dgl.graph((rna_Gaus_edge_index[0], rna_Gaus_edge_index[1]))
    similarity_feature['mm_g'] = {'Data_M': rna_Gaus_sim, 'edges': rna_Gaus_edge_index, 'g': g_mm_g}
    miRNA_similarity = rna_Gaus_sim + rna_seq_sim
    similarity_feature['m_s'] = {'Data_M': miRNA_similarity}

    "disease semantic sim"
    dis_semantic_sim = np.loadtxt(os.path.join(path, 'd_ss.csv'), delimiter=',', dtype=float)
    dis_semantic_sim = torch.tensor(dis_semantic_sim, device=device).to(torch.float32)
    dis_sem_edge_index = get_edge_index(dis_semantic_sim, device)
    g_dd_s = dgl.graph((dis_sem_edge_index[0], dis_sem_edge_index[1]))
    similarity_feature['dd_s'] = {'Data_M': dis_semantic_sim, 'edges': dis_sem_edge_index, 'g': g_dd_s}

    "disease Gaussian sim"
    dis_Gaus_sim = np.loadtxt(os.path.join(path, 'd_gs.csv'), delimiter=',', dtype=float)
    dis_Gaus_sim = torch.tensor(dis_Gaus_sim, device=device).to(torch.float32)
    dis_Gaus_edge_index = get_edge_index(dis_Gaus_sim, device)
    g_dd_g = dgl.graph((dis_Gaus_edge_index[0], dis_Gaus_edge_index[1]))
    similarity_feature['dd_g'] = {'Data_M': dis_Gaus_sim, 'edges': dis_Gaus_edge_index, 'g': g_dd_g}

    disease_similarity = dis_semantic_sim + dis_Gaus_sim
    similarity_feature['d_s'] = {'Data_M': disease_similarity}

    return similarity_feature


# ***************************载入图的边数据开始**************************
def load_fold_data(args):
    path = args.path
    kfolds = args.kfolds
    edge_idx_dict = dict()
    g = dict()
    md_matrix = np.loadtxt(os.path.join(path + '/m_d.csv'), dtype=int, delimiter=',')
    edge_idx_dict['true_md'] = md_matrix
    m_adj = np.load(os.path.join(path, 'm_adj.npy'))
    d_adj = np.load(os.path.join(path, 'd_adj.npy'))

    rng = np.random.default_rng(seed=42)  # 固定训练测试

    pos_samples = np.where(md_matrix == 1)
    pos_samples = (pos_samples[0], pos_samples[1] + md_matrix.shape[0])
    pos_samples_shuffled = rng.permutation(pos_samples, axis=1)

    train_pos_edges = pos_samples_shuffled  # 11201正 90%

    neg_samples = np.where(md_matrix == 0)
    neg_samples = (neg_samples[0], neg_samples[1] + md_matrix.shape[0])
    neg_samples_shuffled = rng.permutation(neg_samples, axis=1)[:, :pos_samples_shuffled.shape[1]]

    train_neg_edges = neg_samples_shuffled

    kf = KFold(n_splits=kfolds, shuffle=True, random_state=1)

    train_pos_edges = train_pos_edges.T
    train_neg_edges = train_neg_edges.T

    train_idx, valid_idx = [], []
    for train_index, valid_index in kf.split(train_pos_edges):
        train_idx.append(train_index)
        valid_idx.append(valid_index)

    for i in range(kfolds):
        edges_train_pos, edges_valid_pos = train_pos_edges[train_idx[i]], train_pos_edges[valid_idx[i]]
        fold_train_pos80 = edges_train_pos.T
        fold_valid_pos20 = edges_valid_pos.T

        edges_train_neg, edges_valid_neg = train_neg_edges[train_idx[i]], train_neg_edges[valid_idx[i]]
        fold_train_neg80 = edges_train_neg.T
        fold_valid_neg20 = edges_valid_neg.T

        fold_100p_100n = np.hstack(
            (np.hstack((fold_train_pos80, fold_valid_pos20)), np.hstack((fold_train_neg80, fold_valid_neg20))))
        fold_train_edges_80p_80n = np.hstack((fold_train_pos80, fold_train_neg80))
        fold_train_label_80p_80n = np.hstack((np.ones(fold_train_pos80.shape[1]), np.zeros(fold_train_neg80.shape[1])))
        fold_valid_edges_20p_20n = np.hstack((fold_valid_pos20, fold_valid_neg20))
        fold_valid_label_20p_20n = np.hstack((np.ones(fold_valid_pos20.shape[1]), np.zeros(fold_valid_neg20.shape[1])))

        edge_idx_dict[str(i)] = {}
        g[str(i)] = {}
        # 可能用到的100
        edge_idx_dict[str(i)]["fold_100p_100n"] = torch.tensor(fold_100p_100n).to(torch.long).to(
            device=args.device)
        g[str(i)]["fold_100p_100n"] = dgl.graph(
            (fold_100p_100n[0], fold_100p_100n[1])).to(device=args.device)
        # 训练用的80
        edge_idx_dict[str(i)]["fold_train_edges_80p_80n"] = torch.tensor(fold_train_edges_80p_80n).to(torch.long).to(
            device=args.device)
        g[str(i)]["fold_train_edges_80p_80n"] = dgl.graph(
            (fold_train_edges_80p_80n[0], fold_train_edges_80p_80n[1])).to(device=args.device)

        edge_idx_dict[str(i)]["fold_train_label_80p_80n"] = torch.tensor(fold_train_label_80p_80n).to(torch.float32).to(
            device=args.device)
        m_adj = torch.tensor(m_adj).to(torch.int64).to(device=args.device)
        d_adj = torch.tensor(d_adj).to(torch.int64).to(device=args.device)

        edge0 = torch.tensor(fold_train_edges_80p_80n[0]).to(device=args.device)
        edge1 = torch.tensor(fold_train_edges_80p_80n[1] - md_matrix.shape[0]).to(device=args.device)

        hete80 = dgl.heterograph({
            ('miRNA', 'm-d', 'disease'): (edge0, edge1),
            ('disease', 'd-m', 'miRNA'): (edge1, edge0),
            ('miRNA', 'm-m', 'miRNA'): (m_adj[0], m_adj[1]),
            ('disease', 'd-d', 'disease'): (d_adj[0], d_adj[1])
        }, device=args.device)
        train_mmdd_meta = []
        train_ddmm_meta = []
        train_mdmd_meta = []
        train_dmdm_meta = []
        for j in range(20):

            mmdd = dgl.sampling.random_walk(hete80, list(range(0, md_matrix.shape[0])), metapath=['m-m', 'm-d', 'd-d'])[
                0].tolist()
            for nodeList in mmdd:
                firstList = nodeList[0]
                for nodeIndex in range(len(nodeList)):
                    if nodeList[nodeIndex] == -1:
                        nodeList[nodeIndex] = firstList
            train_mmdd_meta.append(mmdd)

            ddmm = dgl.sampling.random_walk(hete80, list(range(0, md_matrix.shape[1])), metapath=['d-d', 'd-m', 'm-m'])[
                0].tolist()
            for nodeList in ddmm:
                firstList = nodeList[0]
                for nodeIndex in range(len(nodeList)):
                    if nodeList[nodeIndex] == -1:
                        nodeList[nodeIndex] = firstList
            train_ddmm_meta.append(ddmm)

            mdmd = dgl.sampling.random_walk(hete80, list(range(0, md_matrix.shape[0])), metapath=['m-d', 'd-m', 'm-d'])[
                0].tolist()
            for nodeList in mdmd:
                firstList = nodeList[0]
                for nodeIndex in range(len(nodeList)):
                    if nodeList[nodeIndex] == -1:
                        nodeList[nodeIndex] = firstList
            train_mdmd_meta.append(mdmd)

            dmdm = dgl.sampling.random_walk(hete80, list(range(0, md_matrix.shape[1])), metapath=['d-m', 'm-d', 'd-m'])[
                0].tolist()
            for nodeList in dmdm:
                firstList = nodeList[0]
                for nodeIndex in range(len(nodeList)):
                    if nodeList[nodeIndex] == -1:
                        nodeList[nodeIndex] = firstList
            train_dmdm_meta.append(dmdm)
        g[str(i)]['train_mmdd_meta'] = torch.tensor(train_mmdd_meta, requires_grad=False)
        g[str(i)]['train_ddmm_meta'] = torch.tensor(train_ddmm_meta, requires_grad=False)
        g[str(i)]['train_mdmd_meta'] = torch.tensor(train_mdmd_meta, requires_grad=False)
        g[str(i)]['train_dmdm_meta'] = torch.tensor(train_dmdm_meta, requires_grad=False)
        # 验证用的20
        edge_idx_dict[str(i)]["fold_valid_edges_20p_20n"] = torch.tensor(fold_valid_edges_20p_20n).to(torch.long).to(
            device=args.device)
        g[str(i)]["fold_valid_edges_20p_20n"] = dgl.graph(
            (fold_valid_edges_20p_20n[0], fold_valid_edges_20p_20n[1])).to(device=args.device)
        edge_idx_dict[str(i)]["fold_valid_label_20p_20n"] = torch.tensor(fold_valid_label_20p_20n).to(torch.float32).to(
            device=args.device)

        edge2 = torch.tensor(fold_valid_edges_20p_20n[0]).to(device=args.device)
        edge3 = torch.tensor(fold_valid_edges_20p_20n[1] - md_matrix.shape[0]).to(device=args.device)

        hete20 = dgl.heterograph({
            ('miRNA', 'm-d', 'disease'): (edge2, edge3),
            ('disease', 'd-m', 'miRNA'): (edge3, edge2),
            ('miRNA', 'm-m', 'miRNA'): (m_adj[0], m_adj[1]),
            ('disease', 'd-d', 'disease'): (d_adj[0], d_adj[1])
        })
        valid_mmdd_meta = []
        valid_ddmm_meta = []
        valid_mdmd_meta = []
        valid_dmdm_meta = []
        for jj in range(20):

            mmdd = dgl.sampling.random_walk(hete20, list(range(0, md_matrix.shape[0])),
                                            metapath=['m-m', 'm-d', 'd-d'])[0].tolist()
            for nodeList in mmdd:
                firstList = nodeList[0]
                for nodeIndex in range(len(nodeList)):
                    if nodeList[nodeIndex] == -1:
                        nodeList[nodeIndex] = firstList
            valid_mmdd_meta.append(mmdd)

            ddmm = dgl.sampling.random_walk(hete20, list(range(0, md_matrix.shape[1])),
                                            metapath=['d-d', 'd-m', 'm-m'])[0].tolist()
            for nodeList in ddmm:
                firstList = nodeList[0]
                for nodeIndex in range(len(nodeList)):
                    if nodeList[nodeIndex] == -1:
                        nodeList[nodeIndex] = firstList
            valid_ddmm_meta.append(ddmm)

            mdmd = dgl.sampling.random_walk(hete20, list(range(0, md_matrix.shape[0])),
                                            metapath=['m-d', 'd-m', 'm-d'])[0].tolist()
            for nodeList in mdmd:
                firstList = nodeList[0]
                for nodeIndex in range(len(nodeList)):
                    if nodeList[nodeIndex] == -1:
                        nodeList[nodeIndex] = firstList
            valid_mdmd_meta.append(mdmd)

            dmdm = dgl.sampling.random_walk(hete20, list(range(0, md_matrix.shape[1])),
                                            metapath=['d-m', 'm-d', 'd-m'])[0].tolist()
            for nodeList in dmdm:
                firstList = nodeList[0]
                for nodeIndex in range(len(nodeList)):
                    if nodeList[nodeIndex] == -1:
                        nodeList[nodeIndex] = firstList
            valid_dmdm_meta.append(dmdm)

        g[str(i)]['valid_mmdd_meta'] = torch.tensor(valid_mmdd_meta, requires_grad=False)
        g[str(i)]['valid_ddmm_meta'] = torch.tensor(valid_ddmm_meta, requires_grad=False)
        g[str(i)]['valid_mdmd_meta'] = torch.tensor(valid_mdmd_meta, requires_grad=False)
        g[str(i)]['valid_dmdm_meta'] = torch.tensor(valid_dmdm_meta, requires_grad=False)

    return edge_idx_dict, g
