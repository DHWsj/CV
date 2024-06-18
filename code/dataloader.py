import torch
from torch.utils.data import Dataset
import pandas as pd
import parse
from parse import args
import scipy.sparse as sp
import torch.nn.functional as F

# 对数据集的处理
class MyDataset(Dataset):
    def __init__(self, train_file, valid_file, test_file, device):
        self.device = device
        # train dataset
        train_data = pd.read_table(train_file, header=None, sep=' ')
        train_pos_data = train_data[train_data[2] >= args.offset]
        train_neg_data = train_data[train_data[2] < args.offset]
        self.train_data = torch.from_numpy(train_data.values).to(self.device)
        self.train_pos_user = torch.from_numpy(train_pos_data[0].values).to(self.device)
        self.train_pos_item = torch.from_numpy(train_pos_data[1].values).to(self.device)
        self.train_pos_unique_users = torch.unique(self.train_pos_user)
        self.train_pos_unique_items = torch.unique(self.train_pos_item)
        self.train_neg_user = torch.from_numpy(train_neg_data[0].values).to(self.device)
        self.train_neg_item = torch.from_numpy(train_neg_data[1].values).to(self.device)
        self.train_neg_unique_users = torch.unique(self.train_neg_user)
        self.train_neg_unique_items = torch.unique(self.train_neg_item)
        # valid dataset
        valid_data = pd.read_table(valid_file, header=None, sep=' ')
        valid_pos_data = valid_data[valid_data[2] >= args.offset]
        valid_neg_data = valid_data[valid_data[2] < args.offset]
        self.valid_data = torch.from_numpy(valid_data.values).to(self.device)
        self.valid_pos_user = torch.from_numpy(valid_pos_data[0].values).to(self.device)
        self.valid_pos_item = torch.from_numpy(valid_pos_data[1].values).to(self.device)
        self.valid_pos_unique_users = torch.unique(self.valid_pos_user)
        self.valid_pos_unique_items = torch.unique(self.valid_pos_item)
        self.valid_neg_user = torch.from_numpy(valid_neg_data[0].values).to(self.device)
        self.valid_neg_item = torch.from_numpy(valid_neg_data[1].values).to(self.device)
        self.valid_neg_unique_users = torch.unique(self.valid_neg_user)
        self.valid_neg_unique_items = torch.unique(self.valid_neg_item)
        # test dataset
        test_data = pd.read_table(test_file, header=None, sep=' ')
        test_pos_data = test_data[test_data[2] >= args.offset]
        test_neg_data = test_data[test_data[2] < args.offset]
        self.test_data = torch.from_numpy(test_data.values).to(self.device)
        self.test_pos_user = torch.from_numpy(test_pos_data[0].values).to(self.device)
        self.test_pos_item = torch.from_numpy(test_pos_data[1].values).to(self.device)
        self.test_pos_unique_users = torch.unique(self.test_pos_user)
        self.test_pos_unique_items = torch.unique(self.test_pos_item)
        self.test_neg_user = torch.from_numpy(test_neg_data[0].values).to(self.device)
        self.test_neg_item = torch.from_numpy(test_neg_data[1].values).to(self.device)
        self.test_neg_unique_users = torch.unique(self.test_neg_user)
        self.test_neg_unique_items = torch.unique(self.test_neg_item)
        self.num_users = max([self.train_pos_unique_users.max(),
                              self.train_neg_unique_users.max(),
                              self.valid_pos_unique_users.max(),
                              self.valid_neg_unique_users.max(),
                              self.test_pos_unique_users.max(),
                              self.test_neg_unique_users.max()]).cpu()+1
        self.num_items = max([self.train_pos_unique_items.max(),
                              self.train_neg_unique_items.max(),
                              self.valid_pos_unique_items.max(),
                              self.valid_neg_unique_items.max(),
                              self.test_pos_unique_items.max(),
                              self.test_neg_unique_items.max()]).cpu()+1
        self.num_nodes = self.num_users+self.num_items
        print('users: %d, items: %d.' % (self.num_users, self.num_items))
        print('train: %d pos + %d neg.' % (self.train_pos_user.shape[0], self.train_neg_user.shape[0]))
        print('valid: %d pos + %d neg.' % (self.valid_pos_user.shape[0], self.valid_neg_user.shape[0]))
        print('test: %d pos + %d neg.' % (self.test_pos_user.shape[0], self.test_neg_user.shape[0]))
        #
        self._train_neg_list = None
        self._train_pos_list = None
        self._valid_neg_list = None
        self._valid_pos_list = None
        self._test_neg_list = None
        self._test_pos_list = None
        self._A_pos = None
        self._A_neg = None
        self._degree_pos = None
        self._degree_neg = None
        self._tildeA = None
        self._tildeA_pos = None
        self._tildeA_neg = None
        self._indices = None
        self._paths = None
        self._values = None
        self._counts = None
        self._counts_sum = None
        self._L = None
        self._L_pos = None
        self._L_neg = None
        self._L_eigs = None

    @ property
    def train_pos_list(self):
        if self._train_pos_list is None:
            self._train_pos_list = [list(self.train_pos_item[self.train_pos_user == u].cpu().numpy()) for u in range(self.num_users)]
        return self._train_pos_list

    @ property
    def train_neg_list(self):
        if self._train_neg_list is None:
            self._train_neg_list = [list(self.train_neg_item[self.train_neg_user == u].cpu().numpy()) for u in range(self.num_users)]
        return self._train_neg_list

    @ property
    def valid_pos_list(self):
        if self._valid_pos_list is None:
            self._valid_pos_list = [list(self.valid_pos_item[self.valid_pos_user == u].cpu().numpy()) for u in self.valid_pos_unique_users]
        return self._valid_pos_list

    @ property
    def valid_neg_list(self):
        if self._valid_neg_list is None:
            self._valid_neg_list = [list(self.valid_neg_item[self.valid_neg_user == u].cpu().numpy()) for u in self.valid_pos_unique_users]
        return self._valid_neg_list

    @ property
    def test_pos_list(self):
        if self._test_pos_list is None:
            self._test_pos_list = [list(self.test_pos_item[self.test_pos_user == u].cpu().numpy()) for u in self.test_pos_unique_users]
        return self._test_pos_list

    @ property
    def test_neg_list(self):
        if self._test_neg_list is None:
            self._test_neg_list = [list(self.test_neg_item[self.test_neg_user == u].cpu().numpy()) for u in self.test_pos_unique_users]
        return self._test_neg_list

    # 生成正样本的邻接矩阵。
    @ property
    def A_pos(self):
        if self._A_pos is None:
            self._A_pos = torch.sparse_coo_tensor(
                torch.cat([
                    torch.stack([self.train_pos_user, self.train_pos_item+self.num_users]),
                    torch.stack([self.train_pos_item+self.num_users, self.train_pos_user])], dim=1),
                torch.ones(self.train_pos_user.shape[0]*2).to(parse.device),
                torch.Size([self.num_nodes, self.num_nodes]))
        return self._A_pos

    # 计算正样本邻接矩阵的度矩阵（节点的度）。
    @ property
    def degree_pos(self):
        if self._degree_pos is None:
            self._degree_pos = self.A_pos.sum(dim=1).to_dense()
        return self._degree_pos

    #  归一化的正样本邻接矩阵。通过度矩阵的逆平方根对邻接矩阵进行归一化。
    @ property
    def tildeA_pos(self):
        if self._tildeA_pos is None:
            D = self.degree_pos.float()
            D[D == 0.] = 1.
            D1 = torch.sparse_coo_tensor(
                torch.arange(self.num_nodes, device=parse.device).unsqueeze(0).repeat(2, 1),
                D**(-1/2),
                torch.Size([self.num_nodes, self.num_nodes]))
            D2 = torch.sparse_coo_tensor(
                torch.arange(self.num_nodes, device=parse.device).unsqueeze(0).repeat(2, 1),
                D**(-1/2),
                torch.Size([self.num_nodes, self.num_nodes]))
            self._tildeA_pos = torch.sparse.mm(torch.sparse.mm(D1, self.A_pos), D2)
        return self._tildeA_pos

    # 生成正样本的图拉普拉斯矩阵
    @ property
    def L_pos(self):
        if self._L_pos is None:
            D = torch.sparse_coo_tensor(
                torch.arange(self.num_nodes, device=parse.device).unsqueeze(0).repeat(2, 1),
                torch.ones(self.num_nodes, device=parse.device),
                torch.Size([self.num_nodes, self.num_nodes]))
            self._L_pos = D-self.tildeA_pos
        return self._L_pos

    # 生成负样本的邻接矩阵
    @ property
    def A_neg(self):
        if self._A_neg is None:
            self._A_neg = torch.sparse_coo_tensor(
                torch.cat([
                    torch.stack([self.train_neg_user, self.train_neg_item+self.num_users]),
                    torch.stack([self.train_neg_item+self.num_users, self.train_neg_user])], dim=1),
                torch.ones(self.train_neg_user.shape[0]*2).to(parse.device),
                torch.Size([self.num_nodes, self.num_nodes]))
        return self._A_neg

    # 计算负样本邻接矩阵的度矩阵。
    @ property
    def degree_neg(self):
        if self._degree_neg is None:
            self._degree_neg = self.A_neg.sum(dim=1).to_dense()
        return self._degree_neg

    # 归一化的负样本邻接矩阵。
    @ property
    def tildeA_neg(self):
        if self._tildeA_neg is None:
            D = self.degree_neg.float()
            D[D == 0.] = 1.
            D1 = torch.sparse_coo_tensor(
                torch.arange(self.num_nodes, device=parse.device).unsqueeze(0).repeat(2, 1),
                D**(-1/2),
                torch.Size([self.num_nodes, self.num_nodes]))
            D2 = torch.sparse_coo_tensor(
                torch.arange(self.num_nodes, device=parse.device).unsqueeze(0).repeat(2, 1),
                D**(-1/2),
                torch.Size([self.num_nodes, self.num_nodes]))
            self._tildeA_neg = torch.sparse.mm(torch.sparse.mm(D1, self.A_neg), D2)
        return self._tildeA_neg

    # 生成负样本的图拉普拉斯矩阵
    @ property
    def L_neg(self):
        if self._L_neg is None:
            D = torch.sparse_coo_tensor(
                torch.arange(self.num_nodes, device=parse.device).unsqueeze(0).repeat(2, 1),
                torch.ones(self.num_nodes, device=parse.device),
                torch.Size([self.num_nodes, self.num_nodes]))
            self._L_neg = D-self.tildeA_neg
        return self._L_neg

    # 其中论文的创新点之一
    # 综合正负样本的图拉普拉斯矩阵，按照比例 args.alpha 进行加权平均
    @ property
    def L(self):
        if self._L is None:
            self._L = (self.L_pos+args.alpha*self.L_neg)/(1+args.alpha)
        return self._L

    # 计算图拉普拉斯矩阵的特征值。
    @ property
    def L_eigs(self):
        if self._L_eigs is None:
            if args.eigs_dim == 0:
                self._L_eigs = torch.tensor([]).to(parse.device)
            else:
                _, self._L_eigs = sp.linalg.eigs(
                    sp.csr_matrix(
                        (self.L._values().cpu(), self.L._indices().cpu()),
                        (self.num_nodes, self.num_nodes)),
                    k=args.eigs_dim,
                    which='SR')
                self._L_eigs = torch.tensor(self._L_eigs.real).to(parse.device)
                self._L_eigs = F.layer_norm(self._L_eigs, normalized_shape=(args.eigs_dim,))
        return self._L_eigs

    # 用于在图结构数据中进行采样操作。其主要功能是从图中采样路径，并生成用于训练的输入特征和目标标签。
    def sample(self):
        if self._indices is None:
            self._indices = torch.cat([
                torch.stack([self.train_pos_user, self.train_pos_item+self.num_users]),
                torch.stack([self.train_pos_item+self.num_users, self.train_pos_user]),
                torch.stack([self.train_neg_user, self.train_neg_item+self.num_users]),
                torch.stack([self.train_neg_item+self.num_users, self.train_neg_user])], dim=1)
            self._paths = torch.cat([
                torch.ones(self.train_pos_user.shape).repeat(2),
                torch.zeros(self.train_neg_user.shape).repeat(2)], dim=0).long().to(parse.device)
            sorted_indices = torch.argsort(self._indices[0, :])
            self._indices = self._indices[:, sorted_indices]
            self._paths = self._paths[sorted_indices]
            self._counts = torch.bincount(self._indices[0], minlength=self.num_nodes)
            self._counts_sum = torch.cumsum(self._counts, dim=0)
            d = torch.sqrt(self._counts)
            d[d == 0.] = 1.
            d = 1./d
            self._values = torch.ones(self._indices.shape[1]).to(
                parse.device)*d[self._indices[0]]*d[self._indices[1]]
        res_X, res_Y = [], []  # 用于存储采样结果
        # 用于记录采样过程中的索引
        record_X = []
        X,  Y,  = self._indices,  torch.ones_like(self._paths).long()*2+self._paths
        # 用于记录循环索引，防止在采样过程中出现循环
        loop_indices = torch.zeros_like(Y).bool()
        for hop in range(args.sample_hop):
            loop_indices = loop_indices | (X[0] == X[1])  # 这是一个布尔表达式，用于检查路径的起点和终点是否相同。如果相同，说明当前路径形成了一个环（循环）。
            # 用于从图结构中进行采样。具体来说，它通过多次跳跃（hop）来生成采样的节点对（X）和路径（Y）。
            for i in range(hop % 2, hop, 2):
                loop_indices = loop_indices | (record_X[i][1] == X[1])
            record_X.append(X)
            res_X.append(X[:, ~loop_indices])  # 表示从 X 中筛选出未形成闭环的节点对
            res_Y.append(Y[~loop_indices]-2)  # 表示从 Y 中筛选出未形成闭环的路径，并将其值减去2。
            # 通过累计计数减去随机跳跃步数得到的下一步采样的节点索引。
            next_indices = self._counts_sum[X[1]]-(torch.rand(X.shape[1]).to(parse.device)*self._counts[X[1]]).long()-1
            X = torch.stack([X[0], self._indices[1, next_indices]], dim=0)
            Y = Y*2+self._paths[next_indices]
        return res_X, res_Y


dataset = MyDataset(parse.train_file, parse.valid_file,
                    parse.test_file, parse.device)
