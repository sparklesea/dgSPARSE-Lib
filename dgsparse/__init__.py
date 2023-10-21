import importlib
import os.path as osp
import torch
import dgsparse._C
import dgsparse.spmm
import dgsparse.tensor
import dgsparse.storage
from .spmm import spmm_max, spmm_mean, spmm_sum, spmm_min
from .spmm_group import spmm_sum_group
from .tensor import SparseTensor
from .storage import Storage
from .ftransform import csr2csc
from . import nn  # noqa

__version__ = '0.1'

for library in ['_spconv', '_spmm', '_spmm_group']:
    cuda_spec = importlib.machinery.PathFinder().find_spec(
        f'{library}_cuda', [osp.dirname(__file__)])
    # cpu_spec = importlib.machinery.PathFinder().find_spec(
    #     f'{library}_cpu', [osp.dirname(__file__)])
    spec = cuda_spec  # or cpu_spec
    if spec is not None:
        torch.ops.load_library(spec.origin)
    else:
        raise ImportError(f"Could not find module '{library}_cuda' in "
                          f'{osp.dirname(__file__)}')

cuda_version = dgsparse._C.cuda_version()
if torch.version.cuda is not None and cuda_version != -1:  # pragma: no cover
    if cuda_version < 10000:
        major, minor = int(str(cuda_version)[0]), int(str(cuda_version)[2])
    else:
        major, minor = int(str(cuda_version)[0:2]), int(str(cuda_version)[3])
    t_major, t_minor = [int(x) for x in torch.version.cuda.split('.')]

    if t_major != major:
        raise RuntimeError(
            f'Detected that PyTorch and dgsparse were compiled with '
            f'different CUDA versions. PyTorch has CUDA version '
            f'{t_major}.{t_minor} and dgsparse has CUDA version '
            f'{major}.{minor}. Please reinstall the dgsparse that '
            f'matches your PyTorch install.')

# from .tensor import SparseTensor

__all__ = [
    'spmm_sum', 'spmm_max', 'spmm_min', 'spmm_mean', 'spmm_sum_group',
    'Storage', 'SparseTensor', 'csr2csc'
]
