#ifndef SPMM_CUDA
#define SPMM_CUDA

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda_util.cuh"

template <typename Index, typename DType>
__global__ void
csrspmm_seqreduce_rowbalance_kernel(const Index nr, const Index feature_size,
                          const Index rowPtr[], const Index colIdx[],
                          const DType values[], const DType dnInput[],
                          DType dnOutput[]) {
  Index row_tile = blockDim.y; // 8
  Index subwarp_id = threadIdx.y;
  Index stride = row_tile * gridDim.x; // 8 * (m/8)
  Index row = blockIdx.x * row_tile + subwarp_id;
  Index v_id = (blockIdx.y * blockDim.x) + threadIdx.x;
  dnInput += v_id;
  dnOutput += v_id;
  DType res = 0, val;
  Index col;
  for (; row < nr; row += stride) {
    Index start = __ldg(rowPtr + row);
    Index end = __ldg(rowPtr + row + 1);
    for (Index p = start; p < end; p++) {
      col = __ldg(colIdx + p);
      val = __guard_load_default_one<DType>(values, p);
      res += val * __ldg(dnInput + col * feature_size);
    }
    dnOutput[row * feature_size] = res;
  }
}

template <typename Index, typename DType>
__global__ void
csrspmm_seqreduce_nnzbalance_kernel(const Index nr, const Index feature_size,
                          const Index nnz_, const Index rowPtr[],
                          const Index colIdx[], const DType values[],
                          const DType dnInput[], DType dnOutput[]) {
  Index nnz = nnz_;
  if (nnz < 0)
    nnz = rowPtr[nr];

  Index Nnzdim_thread = blockDim.y * gridDim.x;
  Index NE_PER_THREAD = CEIL(nnz, Nnzdim_thread);
  Index eid = (blockIdx.x * blockDim.y + threadIdx.y) * NE_PER_THREAD;
  Index v_id = (blockIdx.y * blockDim.x) + threadIdx.x;
  Index col = 0;
  DType val = 0.0;
  if(v_id < feature_size)
  {
    if (eid < nnz) {
      Index row = binary_search_segment_number<Index>(rowPtr, nr, nnz, eid);
      Index step = __ldg(rowPtr + row + 1) - eid;

      for (Index ii = 0; ii < NE_PER_THREAD; ii++) {
        if (eid >= nnz)
          break;
        if (ii < step) {
          col = __ldg(colIdx + eid) * feature_size;
          val += __guard_load_default_one<DType>(values, eid) *
                __ldg(dnInput + col + v_id);

          eid++;
        } else {
          atomicAdd(&dnOutput[row * feature_size + v_id], val);

          row = binary_search_segment_number<Index>(rowPtr, nr, nnz, eid);
          step = __ldg(rowPtr + row + 1) - eid;
          col = __ldg(colIdx + eid) * feature_size;
          val = __guard_load_default_one<DType>(values, eid) *
                __ldg(dnInput + col + v_id);

          eid++;
        }
      }
      atomicAdd(&dnOutput[row * feature_size + v_id], val);
    }
  }
}

// Parallel-reduction algorithm assigns a warp to a non-zero segment
//   and use primitives like parallel-reduction / parallel-scan
//   to compute SpMM.
template <typename Index, typename DType, typename access_t>
__global__ void csrspmm_parreduce_rowbalance_kernel(
    const Index nr, const Index feature_size, const Index rowPtr[],
    const Index colIdx[], const DType values[], const DType dnInput[],
    DType dnOutput[]) {
  constexpr Index CoarsenFactor = sizeof(access_t) / sizeof(DType);

  Index lane_id = (threadIdx.x & (32 - 1));
  Index stride = gridDim.x * blockDim.y;
  Index row = blockIdx.x * blockDim.y + threadIdx.y;

  // get the dense column offset
  Index col_offset = blockIdx.y * 32 + (threadIdx.x >> 5) * CoarsenFactor;
  const DType *B_panel = dnInput + col_offset;
  DType *C_panel = dnOutput + col_offset;
  Index ldB = feature_size;
  Index ldC = feature_size;

  if (col_offset >= feature_size)
    return;
  if (col_offset + CoarsenFactor >= feature_size)
    goto Ndim_Residue;

  for (; row < nr; row += stride) {
    // declare accumulators
    DType c[CoarsenFactor] = {0};
    DType buffer[CoarsenFactor];

    Index start = rowPtr[row];
    Index end = rowPtr[row + 1];
    Index k;
    DType v;

    for (Index jj = start + lane_id; jj < end; jj += 32) {
      k = colIdx[jj];
      v = __guard_load_default_one<DType>(values, jj);

      // load B-elements in vector-type
      *(access_t *)buffer = *(access_t *)(B_panel + k * ldB);

#pragma unroll
      for (Index i = 0; i < CoarsenFactor; i++) {
        c[i] += v * buffer[i];
      }
    }

#pragma unroll
    for (Index i = 0; i < CoarsenFactor; i++) {
      // row-wise reduction is a simple merge-tree
      SHFL_DOWN_REDUCE(c[i])
    }

    // store to C in vector-type
    if (lane_id == 0) {
      *(access_t *)(C_panel + row * ldC) = *(access_t *)c;
    }
  }
  return;

Ndim_Residue:
  Index valid_lane_num = feature_size - col_offset;

  for (; row < nr; row += stride) {
    // get row offsets
    DType c[CoarsenFactor] = {0};
    DType buffer[CoarsenFactor];
    // access_t res = init_zeros<access_t>();

    Index start = rowPtr[row];
    Index end = rowPtr[row + 1];
    Index k;
    DType v;

    for (Index jj = start + lane_id; jj < end; jj += 32) {
      k = colIdx[jj];
      v = __guard_load_default_one<DType>(values, jj);

#pragma unroll
      for (Index i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
          buffer[i] = B_panel[k * ldB + i];
        }
      }

#pragma unroll
      for (Index i = 0; i < CoarsenFactor; i++) {
        c[i] += v * buffer[i];
      }
    }

#pragma unroll
    for (Index i = 0; i < CoarsenFactor; i++) {
      SHFL_DOWN_REDUCE(c[i])
    }

    if (lane_id == 0) {
#pragma unroll
      for (Index i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
          C_panel[row * ldC + i] = c[i];
        }
      }
    }
  }
}

template <typename Index, typename DType, typename access_t>
__global__ void
csrspmm_parreduce_nnzbalance_kernel(
                    const Index nr, const Index feature_size, const Index nnz_,
                    const Index rowPtr[], const Index colIdx[], const DType values[],
                    const DType dnInput[], DType dnOutput[]) {
  constexpr Index CoarsenFactor = sizeof(access_t) / sizeof(DType);
  Index nnz = nnz_;
  if (nnz < 0)
    nnz = rowPtr[nr];

  Index lane_id = (threadIdx.x & (32 - 1));
  Index Nnzdim_warp_id = blockIdx.x * blockDim.y + threadIdx.y;
  Index nz_start = Nnzdim_warp_id * 32;
  Index stride = gridDim.x * (blockDim.y * 32);

  // get the dense column offset
  Index col_offset = blockIdx.y * 32 + (threadIdx.x >> 5) * CoarsenFactor;
  const DType *B_panel = dnInput + col_offset;
  DType *C_panel = dnOutput + col_offset;
  Index ldB = feature_size;
  Index ldC = feature_size;

  Index k;
  DType v;
  DType c[CoarsenFactor] = {0};
  DType buffer[CoarsenFactor] = {0};

  if (col_offset >= feature_size)
    return;
  if (col_offset + CoarsenFactor >= feature_size)
    goto Ndim_Residue;

  for (int nz_id = nz_start + lane_id;
       nz_id < nnz + lane_id; // make sure NO warp loop-divergence
       nz_id += stride) {
    Index row = binary_search_segment_number<Index>(rowPtr, nr, nnz, nz_id);

    if (nz_id < nnz) {
      k = colIdx[nz_id];
      v = __guard_load_default_one<DType>(values, nz_id);
    } else {
      k = 0;
      v = 0.0f;
    }

    // load B-elements in vector-type
    *(access_t *)buffer = *(access_t *)(B_panel + k * ldB);
#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      c[i] = buffer[i] * v;
    }

    // reduction
    Index row_intv =
        __shfl_sync(FULLMASK, row, (32 - 1)) - __shfl_sync(FULLMASK, row, 0);
    if (row_intv == 0) {
// if all non-zeros in this warp belong to the same row, use a simple reduction
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        SHFL_DOWN_REDUCE(c[i]);
      }
      if (lane_id == 0) {
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          atomicAdd(C_panel + row * ldC + i, c[i]);
        }
      }
    } else {
      // if non-zeros belong to different rows, use a parallel-scan primitive
      // thread that holds the start of each segment are responsible for writing
      // results
      bool is_seg_start =
          ((__shfl_up_sync(FULLMASK, row, 1) != row) || (lane_id == 0));
      DType tmpv;
      Index tmpr;
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        SEG_SHFL_SCAN(c[i], tmpv, row, tmpr);
      }
      if (is_seg_start) {
// atomic add has no vector-type form.
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          atomicAdd(C_panel + row * ldC + i, c[i]);
        }
      }
    }
  }
  return;
Ndim_Residue:
  Index valid_lane_num = feature_size - col_offset;

  for (int nz_id = nz_start + lane_id;
       nz_id < nnz + lane_id; // make sure NO warp loop-divergence
       nz_id += stride) {
    Index row = binary_search_segment_number<Index>(rowPtr, nr, nnz, nz_id);

    if (nz_id < nnz) {
      k = colIdx[nz_id];
      v = __guard_load_default_one<DType>(values, nz_id);
    } else {
      k = 0;
      v = 0.0f;
    }

#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      if (i < valid_lane_num) {
        c[i] = B_panel[k * ldB + i] * v;
      }
    }

    // reduction
    Index row_intv =
        __shfl_sync(FULLMASK, row, (32 - 1)) - __shfl_sync(FULLMASK, row, 0);
    if (row_intv == 0) {
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        SHFL_DOWN_REDUCE(c[i]);
      }
      if (lane_id == 0) {
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          if (i < valid_lane_num) {
            atomicAdd(C_panel + row * ldC + i, c[i]);
          }
        }
      }
    } else {
      bool is_seg_start =
          ((__shfl_up_sync(FULLMASK, row, 1) != row) || (lane_id == 0));
      DType tmpv;
      Index tmpr;
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        SEG_SHFL_SCAN(c[i], tmpv, row, tmpr);
      }
      if (is_seg_start) {
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          if (i < valid_lane_num) {
            atomicAdd(C_panel + row * ldC + i, c[i]);
          }
        }
      }
    }
  }
  return;
}

#endif
