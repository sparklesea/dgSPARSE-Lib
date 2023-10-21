#ifndef SPMM_CUDA
#define SPMM_CUDA

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "../gspmm.h"
#include "cuda_util.cuh"

template <typename Index, typename DType>
__global__ void csrspmm_neighbor_group_kernel(
    const Index edge_groups, const Index feature_size, const Index group_key[],
    const Index group_row[], const Index colIdx[], const DType values[],
    const DType dnInput[], DType dnOutput[], Index E[]) {
  Index group_tile = blockDim.y; // combine a set of groups together
  Index subwarp_id = threadIdx.y;
  Index group = blockIdx.x * group_tile + subwarp_id; // which node_group
  Index v_id = threadIdx.x;
  if (group < edge_groups) {
    Index row = group_row[group]; // get the specific row of each node group
    dnInput += v_id;
    dnOutput += v_id;
    DType res = 0, val;
    Index col;
    Index start = __ldg(group_key + group);
    Index end = __ldg(group_key + group + 1);
    for (Index p = start; p < end; p++) {
      DType val_pre_red = 0;
      col = __ldg(colIdx + p);
      val = __guard_load_default_one<DType>(values, p);
      val_pre_red = val * __ldg(dnInput + col * feature_size);
      res += val_pre_red;
    }
    atomicAdd(dnOutput + row * feature_size,
              res); // atomic, cuz different node group -> same row
  }
}

#endif
