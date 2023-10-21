import torch
from dgsparse import spmm_sum_group
from dgsparse import SparseTensor
import pytest
from utils import GraphDataset

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class SpMMSum:

    def __init__(self, data, in_dim, device, group_size) -> None:
        # prepare for torch and dgsparse
        self.tcsr = data.tcsr

        self.dcsr = SparseTensor.from_torch_sparse_csr_tensor_with_group(
            self.tcsr.clone().detach(),
            True,
            requires_grad=True,
            group_size=group_size)

        self.in_dim = in_dim
        self.device = device
        self.input_feature = torch.rand((data.num_nodes, in_dim),
                                        requires_grad=True,
                                        device=device)

    def forward_check(self):
        out_check = torch.sparse.mm(self.tcsr, self.input_feature)
        out = spmm_sum_group(self.dcsr, self.input_feature)
        print(torch.max(torch.abs(out - out_check)))
        assert torch.allclose(out, out_check, atol=1e-3)

    def backward_check(self):
        out_check = torch.sparse.mm(self.tcsr, self.input_feature)
        out_check.sum().backward()
        dX_check = self.input_feature.grad
        dA_check = self.tcsr.grad
        out = spmm_sum_group(self.dcsr, self.input_feature)
        out.sum().backward()
        dX = self.input_feature.grad
        dA_nnz = self.dcsr.storage._values.grad

        print(torch.max(torch.abs(dA_nnz - dA_check.values())))

        assert torch.allclose(dX, dX_check)
        assert torch.allclose(dA_nnz, dA_check.values(), atol=1e-3)


datasets = ['cora', 'citeseer', 'pubmed', 'ppi']
features = [32, 64, 128]
# group_size = [4, 8, 16, 32]
group_size = [1, 2, 4, 8, 16]


@pytest.mark.parametrize('dataset', datasets)
@pytest.mark.parametrize('feat', features)
@pytest.mark.parametrize('gsize', group_size)
def test_spmm_sum(dataset, feat, gsize):
    data = GraphDataset(dataset, 0)
    gc = SpMMSum(data, feat, 0, gsize)
    gc.forward_check()
    gc.backward_check()


# for gsize in group_size:
#     test_spmm_sum('pubmed', 128, gsize)
#     print('group_size: ' + str(gsize) + ' pass')
