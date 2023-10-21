import torch_geometric.datasets as datasets
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_scipy_sparse_matrix
import torch
# import torch_sparse
import scipy.io as sio
import numpy as np


def dataset_transform(name):
    if name == 'arxiv':
        arxiv = PygNodePropPredDataset(root='./data/', name='ogbn-arxiv')
        graph = arxiv[0]
    elif name == 'proteins':
        proteins = PygNodePropPredDataset(root='./data/', name='ogbn-proteins')
        graph = proteins[0]
    elif name == 'products':
        products = PygNodePropPredDataset(root='./data/', name='ogbn-products')
        graph = products[0]
    elif name == 'pubmed':
        dataset = datasets.Planetoid(root='./data/', name='Pubmed')
        graph = dataset[0]
    elif name == 'citeseer':
        dataset = datasets.Planetoid(root='./data/', name='Citeseer')
        graph = dataset[0]
    elif name == 'cora':
        dataset = datasets.Planetoid(root='./data/', name='Cora')
        graph = dataset[0]
    elif name == 'ppi':
        dataset = datasets.PPI(root='./data/Ppi')
        graph = dataset[0]
    elif name == 'reddit':
        dataset = datasets.Reddit(root='./data/Reddit')
        graph = dataset[0]
    elif name == 'github':
        dataset = datasets.GitHub(root='./data')
        graph = dataset[0]
    else:
        raise KeyError('Unknown dataset {}.'.format(name))
    sparse_matrix = sio.mmread('./pubmed.mtx')
    row = np.concatenate((sparse_matrix.row, sparse_matrix.col), axis=0)
    col = np.concatenate((sparse_matrix.col, sparse_matrix.row), axis=0)
    edge_index = torch.stack((torch.from_numpy(row), torch.from_numpy(col)),
                             dim=1)
    scipy_coo = to_scipy_sparse_matrix(edge_index.T, num_nodes=graph.num_nodes)
    scipy_csr = scipy_coo.tocsr()
    print(scipy_csr)
    sio.mmwrite(f'./{name}_sym.mtx', scipy_csr)


if __name__ == '__main__':
    dataset_transform('pubmed')
