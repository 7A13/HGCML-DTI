import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score,  precision_recall_curve
from sklearn.metrics import auc as auc3
from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio
from sklearn.metrics.pairwise import cosine_similarity as cos
import time
import scipy.spatial.distance as dist
from CLaugmentdti import *


def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


default_configure = {
    'batch_size': 20
}

heter_configure = {
    "lr": 0.0001,
    "dropout": 0,
    "cl_loss_co": 0.5,
    "reg_co": 0.0003,
    "in_size": 512,
    "hidden_size": 256,
    "out_size": 128,
    "weight_decay": 1e-10

}
Es_configure = {
    "lr": 0.0001,
    "dropout": 0,
    "cl_loss_co": 0.5,
    "reg_co": 0.0003,
    "in_size": 512,
    "hidden_size": 256,
    "out_size": 128,
    "weight_decay": 1e-10

}
ICs_configure = {
    "lr": 0.0001,
    "dropout": 0,
    "cl_loss_co": 0.5,
    "reg_co": 0.0003,
    "in_size": 512,
    "hidden_size": 256,
    "out_size": 128,
    "weight_decay": 1e-10

}

Zheng_configure = {
    "lr": 0.0005,
    "dropout": 0.4,
    "cl_loss_co": 0.5,
    "reg_co": 0.0003,
    "in_size": 512,
    "hidden_size": 256,
    "out_size": 128,
    "weight_decay": 1e-10

}


def setup(args,seed):
    args.update(default_configure)
    set_random_seed(seed)
    return args


def comp_jaccard(M):
    matV = np.mat(M)
    x = dist.pdist(matV, 'jaccard')

    k = np.eye(matV.shape[0])
    count = 0
    for i in range(k.shape[0]):
        for j in range(i + 1, k.shape[1]):
            k[i][j] = x[count]
            k[j][i] = x[count]
            count += 1
    return k


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()




def load_graph(feature_edges, n):
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sparse.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(n, n),
                             dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    #nfadj = fadj + sparse.eye(fadj.shape[0])
    return torch.Tensor(fadj.A)

    # nfadj = normalize(fadj + sparse.eye(fadj.shape[0]))
    # nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    # return nfadj



def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def construct_fgraph(features, topk):
    ##### Kernel
    # dist = -0.5 * pair(features) ** 2
    # dist = np.exp(dist)

    #### Cosine
    dist = cos(features)
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)
    edge = []
    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                edge.append([i, vv])
    return edge
def generate_knn(data):
    topk = 17

    edge = construct_fgraph(data, topk)
    res = []

    for line in edge:
        start, end = line[0], line[1]
        if int(start) < int(end):
            res.append([start, end])
    return res


def constructur_graph(dateset, h1, h2, edge, aug=False):

    feature = torch.cat((h1[dateset[:, :1]], h2[dateset[:, 1:2]]), dim=2)

    feature = feature.squeeze(1)
    #edge = np.loadtxt(r"F:\Desktop\HPN\data\zeng\dtiedge.txt", dtype=int)
    # for i in range(dateset.shape[0]):
    #     for j in range(i, dateset.shape[0]):
    #         if dateset[i][0] == dateset[j][0] or dateset[i][1] == dateset[j][1]:
    #             edge.append([i, j])
    # fedge = np.array(generate_knn(feature.cpu().detach().numpy()))

    # if aug:
    #     edge_aug = aug_random_edge(np.array(edge))
    #     edge_aug = load_graph(np.array(edge_aug), dateset.shape[0])
    #     edge = load_graph(np.array(edge), dateset.shape[0])
    #
    #     feature_aug = aug_random_mask(feature)
    #     return edge, feature, edge_aug, feature_aug
    edge = load_graph(np.array(edge), dateset.shape[0])
    # print(edge)
    # a=torch.sum(edge, 0)
    # print(a)
    # j=0
    # for i in range(len(edge)):
    #     if a[i]<=3:
    #         j=j+1
    #         print(a[i])
    #         print(i)
    # print("j",j)
    # wad

    return edge, feature


def constructure_knngraph(dateset, h1, h2, aug=False):
    feature = torch.cat((h1[dateset[:, :1]], h2[dateset[:, 1:2]]), dim=2)
    feature = feature.squeeze(1)


    fedge = np.array(generate_knn(feature.cpu().detach().numpy()))

    if aug:
        fedge_aug = aug_random_edge(np.array(fedge))
        feature_aug = aug_random_mask(feature)
        fedge_aug = load_graph(np.array(fedge_aug), dateset.shape[0])
        fedge = load_graph(np.array(fedge), dateset.shape[0])

        return fedge, feature, fedge_aug, feature_aug
    else:
        fedge = load_graph(np.array(fedge), dateset.shape[0])

        return fedge, feature


def get_clGraph(data, cledg):
    # cledg = np.loadtxt(f"{task}_cledge.txt", dtype=int)
    # cledg = np.loadtxt(r"F:\Desktop\HPN\data\zeng\dti_cledge.txt", dtype=int)
    cl = torch.eye(len(data))
    for i in cledg:
        cl[i[0]][i[1]] = 1
    return cl


def get_set(data, split=5):
    """
    :param data: dataset and label
    :return:
    testset index and trainset index
    """
    set1 = []
    set2 = []
    skf = StratifiedKFold(n_splits=split, shuffle=True)
    for train_index, test_index in skf.split(data[:, :2], data[:, 2:3]):
        set1.append(train_index)
        set2.append(test_index)
    return set1[0].reshape(-1), set2[0].reshape(-1)
def get_cross(data, split=5):
    """
    :param data: dataset and label
    :return:
    testset index and trainset index
    """
    set1 = []
    set2 = []
    skf = StratifiedKFold(n_splits=split, shuffle=True)
    for train_index, test_index in skf.split(data[:, :2], data[:, 2:3]):
        set1.append(train_index)
        set2.append(test_index)
    return set1, set2


def get_roc(out, label):
    return roc_auc_score(label.cpu(), out[:, 1:].cpu().detach().numpy())
def get_pr(out, label):
    precision, recall, thresholds = precision_recall_curve(label.cpu(), out[:, 1:].cpu().detach().numpy())
    return auc3(recall, precision)


def get_f1score(out, label):
    return f1_score(label.cpu(), out.argmax(dim=1).cpu().detach().numpy())


def get_L2reg(parameters):
    reg = 0
    for param in parameters:
        reg += 0.5 * (param ** 2).sum()
    return reg


