#include <torch/extension.h>

#include <tuple>
#include <vector>

#include "../gspmm.h"

std::vector<torch::Tensor>
spmm_group_cuda(torch::Tensor csrptr, torch::Tensor indices,
                torch::Tensor edge_val, torch::Tensor in_feat,
                torch::Tensor group_key, torch::Tensor group_row,
                bool has_value);

torch::Tensor sddmm_cuda_csr(torch::Tensor rowptr, torch::Tensor colind,
                             torch::Tensor D1, torch::Tensor D2);

std::vector<torch::Tensor> csr2csc_cuda(torch::Tensor csrRowPtr,
                                        torch::Tensor csrColInd,
                                        torch::Tensor csrVal);
