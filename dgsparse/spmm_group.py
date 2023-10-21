import torch
from dgsparse.tensor import SparseTensor

# torch.ops.load_library("_spmm_cuda.so")

# torch.ops.dgsparse.SpMM


def spmm_sum_group(sparse: SparseTensor, dense: torch.Tensor) -> torch.Tensor:
    r"""
    Matrix multiplication of a sparse tensor
    and a dense tensor with sum reduction.

    Args:
        sparse (SparseTensor): The sparse tensor.
        dense (Tensor): The dense tensor.

    rtype: :class:'Tensor'
    """
    has_value = sparse.has_value
    rowptr = sparse.storage._rowptr
    col = sparse.storage._col
    values = sparse.storage._values
    group_rowptr = sparse.storage._group_rowptr
    group_row = sparse.storage._group_row
    group_colptr = sparse.storage._group_colptr
    group_col = sparse.storage._group_col

    return torch.ops.dgsparse_spmm_group.spmm_sum_group(
        rowptr, col, values, dense, group_rowptr, group_row, group_colptr,
        group_col, has_value)
