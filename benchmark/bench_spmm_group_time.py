import torch
from dgsparse import spmm_sum_group
from dgsparse import SparseTensor
# import pytest
# import torch_sparse
import time
import dgl
# import tqdm
from utils import GraphDataset


class SpMMSum:

    def __init__(self, data, in_dim, device, group_size) -> None:
        # prepare for torch and dgsparse
        self.tcsr = data.tcsr
        self.dcsr = SparseTensor.from_torch_sparse_csr_tensor_with_group(
            self.tcsr.clone().detach(),
            True,
            requires_grad=True,
            group_size=group_size)
        self.adj_t = data.adj_t
        # self.dgl_A = data.dgl_A
        self.dgl_graph = data.dgl_graph

        self.device = device
        # self.input_feature = data.features
        self.input_feature = torch.rand((data.num_nodes, in_dim),
                                        device=device)
        # self.input_feature =
        # torch.randn((self.dcsr.storage.sparse_sizes[1], 1)).to(device)
        # print(self.input_feature.size())

    def forward_check(self):
        # warm up
        for _ in range(10):
            # out = matmul(self.adj_t, self.input_feature, reduce="sum")
            self.adj_t.spmm(self.input_feature, reduce='sum')
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            # out = matmul(self.adj_t, self.input_feature, reduce="sum")
            self.adj_t.spmm(self.input_feature, reduce='sum')
        torch.cuda.synchronize()
        end = time.time()
        torch_sparse_time = end - start

        # warm up
        for _ in range(10):
            dgl.ops.copy_u_sum(self.dgl_graph, self.input_feature)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            dgl.ops.copy_u_sum(self.dgl_graph, self.input_feature)
        torch.cuda.synchronize()
        end = time.time()
        dgl_time = end - start

        # warm up
        for _ in range(10):
            spmm_sum_group(self.dcsr, self.input_feature)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            spmm_sum_group(self.dcsr, self.input_feature)
        torch.cuda.synchronize()
        end = time.time()
        dgsparse_time = end - start

        return torch_sparse_time, dgl_time, dgsparse_time

    def backward_check(self):
        pass


def check_time(gc, direction='forward'):
    print(f'{direction} time:')
    torch_sparse_time_list = []
    dgl_time_list = []
    dgsparse_time_list = []
    if direction == 'forward':
        torch_sparse_time, dgl_time, dgsparse_time = gc.forward_check()
    elif direction == 'backward':
        torch_sparse_time, dgsparse_time = gc.backward_check()
    else:
        raise ValueError
    torch_sparse_time_list.append(torch_sparse_time)
    dgl_time_list.append(dgl_time)
    dgsparse_time_list.append(dgsparse_time)
    print(f'torch_sparse forward time is: {torch_sparse_time_list}')
    print(f'dgl forward time is: {dgl_time_list}')
    print(f'dgsparse forward time is: {dgsparse_time_list}')


def test_spmm_time(dataset, in_dim, device, reduce='sum'):
    print()
    print(f'start testing {dataset} dataset, \
        reduce is: {reduce}, in_dim is: {in_dim}')
    data = GraphDataset(dataset, device)
    if reduce == 'sum':
        gc = SpMMSum(data, in_dim, device, 4)
    else:
        raise ValueError
    check_time(gc, direction='forward')
    # check_time(gc, direction="backward")


if __name__ == '__main__':
    # datasets = ["cora"]
    device = 'cuda:7' if torch.cuda.is_available() else 'cpu'
    datasets = ['cora', 'citeseer', 'pubmed', 'ppi', 'reddit']
    features_dim = [32, 64, 128]
    for dataset in datasets:
        for in_dim in features_dim:
            test_spmm_time(dataset, in_dim, device, reduce='sum')
