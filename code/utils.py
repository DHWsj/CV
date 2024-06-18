import parse
import numpy as np
import torch
import scipy.sparse as sp
import torch.nn.functional as F
from parse import args


# 用于计算推荐系统的二进制匹配标签和召回率所需的相关数据
def getlabel(test_data, pred_data):
    r, recall_n = [], []
    for i in range(len(pred_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        if len(groundTrue) > 0:
            r.append(list(map(lambda x: x in groundTrue, predictTopK)))
            recall_n.append(len(groundTrue))
    return np.array(r), recall_n


# 用于评估推荐系统的性能指标，包括准确率（Precision）、召回率（Recall）和归一化折损累积增益（NDCG）。
def test(sorted_items, groundTrue):
    sorted_items = sorted_items.cpu().numpy()
    r, recall_n = getlabel(groundTrue, sorted_items)
    pre, recall, ndcg, ndcg2 = [], [], [], []
    for k in parse.topks:
        now_k = min(k, r.shape[1])
        pred = r[:, :now_k]
        right_pred = pred.sum(1)
        # precision
        pre.append(np.sum(right_pred / now_k))
        # recall
        recall.append(np.sum(right_pred / recall_n))
        # ndcg
        dcg = np.sum(pred * (1. / np.log2(np.arange(2, now_k + 2))), axis=1)  # 累积增益
        d_val = [np.sum(1. / np.log2(np.arange(2, i + 2)))
                 for i in range(0, now_k + 1)]
        idcg = np.array([d_val[int(i)] for i in np.minimum(recall_n, now_k)])  # 理想累积增益
        ndcg.append(np.sum(dcg / idcg))
    return torch.tensor(pre), torch.tensor(recall), torch.tensor(ndcg)


# 给定值进行归一化
def sum_norm(indices, values, n):
    s = torch.zeros(n, device=values.device).scatter_add(0, indices[0], values)
    s[s == 0.] = 1.
    return values / s[indices[0]]


# 处理稀疏矩阵进行softmax
def sparse_softmax(indices, values, n):
    return sum_norm(indices, torch.clamp(torch.exp(values), min=-5, max=5), n)
# torch.clamp(torch.exp(values), min=-5, max=5) 这行代码的实际作用是：
# 对 values 中每个元素进行指数变换，得到 exp_values。
# 将 exp_values 中超过 5 的值裁剪为 5，保留范围内的值不变。
# 这样可以确保经过指数变换后的值不会过大，从而避免数值不稳定或溢出问题。
