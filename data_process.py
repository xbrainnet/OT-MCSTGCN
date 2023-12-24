import os
import random
from collections import Counter

import networkx as nx
import numpy as np
import ot
import torch
from scipy import interpolate
# from numpy import random
from scipy.io import loadmat
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.special import chebyt

# from k_order import  chebyshev_polynomials
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

seed = 7
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


# 82,76,68
def MaxMinNormalization(x, Max, Min):
    x = (x - Min) / (Max - Min)
    return x


# ########################################classificatin task: NC  vs. SMC&EMCI########################################
# from torch.utils.data import Dataset, DataLoader
#
# m = loadmat('.\datasets\ADNI_fmri.mat')  # fmri
# keysm = list(m.keys())
# fdata = m[keysm[3]]  #Time series signal
# labels = m[keysm[4]][0]  # Counter({0: 114, 1: 103, 2: 89})
# for i in range(fdata.shape[0]):
#     if labels[i] == 2:
#         labels[i] = 1
# print(labels)
# for i in range(fdata.shape[0]):
#     max_t = np.max(fdata[i])
#     min_t = np.min(fdata[i])
#     fdata[i] = MaxMinNormalization(fdata[i], max_t, min_t)
#
# n = loadmat('.\datasets\ADNI_DTI.mat')  # DTI
# keysn = list(n.keys())
# ddata = n[keysn[3]]  #Structural brain network


# ########################################classificatin task: NC vs. SMC########################################
# from torch.utils.data import Dataset, DataLoader
# m = loadmat('.\datasets\ADNI_fmri.mat')  # fmri
# keysm = list(m.keys())
# fdata = m[keysm[3]][0:158]#Time series signal
# labels = m[keysm[4]][0][0:158]  # Counter({0: 114, 1: 103, 2: 89})
# for i in range(labels.shape[0]):
#     if labels[i] == 2:
#         labels[i] = 1
# print(len(labels))
# for i in range(158):
#     max_t = np.max(fdata[i])
#     min_t = np.min(fdata[i])
#     fdata[i] = MaxMinNormalization(fdata[i], max_t, min_t)
#
#
# n = loadmat('.\datastes\ADNI_DTI.mat')  # DTI
# keysn = list(n.keys())
# ddata = n[keysn[3]][0:158]#Structural brain network


# ########################################classificatin task: NC vs. EMCI########################################
# from torch.utils.data import Dataset, DataLoader
#
# m = loadmat('.\datasets\ADNI_fmri.mat')  # fmri
# keysm = list(m.keys())
# fdata = m[keysm[3]]
# fdata = torch.cat((torch.tensor(fdata[0:82]), torch.tensor(fdata[158:226])))
# fdata = fdata.numpy()#Time series signal
# labels = m[keysm[4]][0]  # Counter({0: 114, 1: 103, 2: 89})
# labels = torch.cat((torch.tensor(labels[0:82]), torch.tensor(labels[158:226])))
# labels = labels.numpy()
# for i in range(fdata.shape[0]):
#     if labels[i] == 2:
#         labels[i] = 1
# for i in range(fdata.shape[0]):
#     max_t = np.max(fdata[i])
#     min_t = np.min(fdata[i])
#     fdata[i] = MaxMinNormalization(fdata[i], max_t, min_t)
#
# n = loadmat('.\datasets\ADNI_DTI.mat')  # DTI
# keysn = list(n.keys())
# ddata = n[keysn[3]]
# ddata = torch.cat((torch.tensor(ddata[0:82]), torch.tensor(ddata[158:226])))
# ddata = ddata.numpy() #Structural brain network

###################################### classificatin task: SMC  vs   EMCI########################################
from torch.utils.data import Dataset, DataLoader

m = loadmat('.\datasets\ADNI_fmri.mat')
keysm = list(m.keys())
fdata = m[keysm[3]][82:226]  # Time series signal
print(fdata.shape)

labels = m[keysm[4]][0][82:226]
for i in range(fdata.shape[0]):
    if labels[i] == 2:
        labels[i] = 0
fdata[np.isnan(fdata)] = -1
for i in range(fdata.shape[0]):
    max_t = np.max(fdata[i])
    min_t = np.min(fdata[i])
    fdata[i] = MaxMinNormalization(fdata[i], max_t, min_t)

n = loadmat('.\datasets\ADNI_DTI.mat')  # DTI
keysn = list(n.keys())
ddata = n[keysn[3]][82:226]  # Structural brain network


index = [i for i in range(fdata.shape[0])]
np.random.shuffle(index)
fdata = fdata[index]
labels = labels[index]
ddata = ddata[index]

"""
DFBNs are constructed using non-overlapping sliding Windows
Input: fMRI, num_window,alpha
Output: DFBNs
"""


def create_DFCN(dataset, num_window, alpha):
    nets_all = []
    win_length = dataset.shape[2] // num_window
    for i in range(dataset.shape[0]):
        nets = []
        datas = dataset[i]  # 90*240
        for j in range(num_window):
            window = datas[:, win_length * j:win_length * (j + 1)]
            net = np.corrcoef(window)
            net[np.isnan(net)] = 0
            nets.append(net)
        nets_all.append(nets)
    nets_all = np.array(nets_all)
    for i in range(nets_all.shape[0]):
        for j in range(nets_all.shape[1]):
            # net1 = dataset[i][j]
            for k in range(nets_all[i][j].shape[0]):
                for m in range(nets_all[i][j].shape[1]):
                    if nets_all[i][j][k][m] <= 0:
                        nets_all[i][j][k][m] = -nets_all[i][j][k][m]

                    if nets_all[i][j][k][m] <= alpha:
                        nets_all[i][j][k][m] = 0
    return nets_all  # torch.Size([306, 4, 90, 90])


"""
DFBNs are constructed using overlapping sliding Windows
Input: fMRI, num_window,alpha
Output: DFBNs
"""
# def create_DFCN(dataset, num_window, alpha):
#     nets_all = []
#     win_length = 240 - num_window * 10
#     for i in range(dataset.shape[0]):
#         nets = []
#         datas = dataset[i]  # 90*240
#         for j in range(num_window):
#             window = datas[:, j * 10: j * 10 + win_length]
#             # print(window.shape)
#             net = np.corrcoef(window)
#             nets.append(net)
#         nets_all.append(nets)
#     nets_all = np.array(nets_all)
#     for i in range(nets_all.shape[0]):
#         for j in range(nets_all.shape[1]):
#             # net1 = dataset[i][j]
#             for k in range(nets_all[i][j].shape[0]):
#                 for m in range(nets_all[i][j].shape[1]):
#                     if nets_all[i][j][k][m] <= 0:
#                         nets_all[i][j][k][m] = -nets_all[i][j][k][m]
#
#                     if nets_all[i][j][k][m] <= alpha:
#                         nets_all[i][j][k][m] = 0
#     return nets_all  # torch.Size([306, 4, 90, 90])


"""
HOTENs are constructed using OT
Input: DFBNs, cost
Output: HOTENs
"""


def creat_ot_briannet(dataset, cost):
    for i in range(dataset.shape[0]):
        A = dict()
        start = []
        for a in range(90):
            A[a] = 1 / 90
        start.append(np.array(list(A.values())))

        for j in range(dataset.shape[1]):
            net = dataset[i][j]
            net = nx.from_numpy_matrix(net)
            graph1 = nx.DiGraph(net)
            graph1.remove_edges_from(nx.selfloop_edges(graph1))
            im = nx.pagerank(graph1, alpha=0.85)
            start.append(np.array(list(im.values())))
            dataset[i][j] = ot.sinkhorn(np.array(start[j]), np.array(start[j + 1]), cost[i], 0.01)
            max_n = np.max(dataset[i][j])
            min_n = np.min(dataset[i][j])
            dataset[i][j] = MaxMinNormalization(dataset[i][j], max_n, min_n)

    return dataset


"""
Construct K-order Chebyshev polynomials 
Input: Structural brain network, k-order
Output: K-order Chebyshev polynomials
"""


def chebyshev_polynomials(adj_matrix, k):
    adj_normalized = 2 * adj_matrix / adj_matrix.max() - 1
    cheb_polynomials = []
    cheb_polynomials.append(np.eye(adj_matrix.shape[0]))
    cheb_polynomials.append(adj_normalized)
    for i in range(2, k + 1):
        next_polynomial = 2 * adj_normalized * cheb_polynomials[i - 1] - cheb_polynomials[i - 2]
        cheb_polynomials.append(next_polynomial)
    cheb_polynomials = np.array(cheb_polynomials)
    return cheb_polynomials


def snet(ddata, k):
    alls = []
    for i in range(ddata.shape[0]):
        data = ddata[i]
        data_cheb = chebyshev_polynomials(data, k)
        alls.append(data_cheb)
    alls = np.array(alls)
    return alls


nets_all = create_DFCN(fdata, 6, 0.65)
ot_net_all = creat_ot_briannet(nets_all, ddata)
chebs = snet(ddata, 2)




class ADNI(Dataset):
    def __init__(self):
        super(ADNI, self).__init__()
        self.ot_nets = ot_net_all
        self.chebs = chebs
        self.label = labels

    def __getitem__(self, item):
        ot_net = self.ot_nets[item]
        cheb = self.chebs[item]
        label = self.label[item]
        return ot_net, cheb, label

    def __len__(self):
        return self.ot_nets.shape[0]
