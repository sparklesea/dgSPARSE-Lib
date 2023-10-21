// #include "../include/cpu/spmm_cpu.h"
#include <torch/all.h>
#include <torch/extension.h>
#include <torch/python.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <vector>

#include "../include/cuda/spmm_group_cuda.h"

torch::Tensor spmm_sum_group(torch::Tensor rowptr, torch::Tensor col,
                             torch::Tensor values, torch::Tensor dense,
                             torch::Tensor group_rowptr,
                             torch::Tensor group_row,
                             torch::Tensor group_colptr,
                             torch::Tensor group_col, bool has_value);

class SpMMSum : public torch::autograd::Function<SpMMSum> {
public:
  static torch::Tensor
  forward(torch::autograd::AutogradContext *ctx, torch::Tensor rowptr,
          torch::Tensor col, torch::Tensor values, torch::Tensor dense,
          torch::Tensor group_rowptr, torch::Tensor group_row,
          torch::Tensor group_colptr, torch::Tensor group_col, bool has_value) {
    auto out = spmm_group_cuda(rowptr, col, values, dense, group_rowptr,
                               group_row, has_value);
    ctx->saved_data["has_value"] = has_value;
    ctx->save_for_backward(
        {rowptr, col, values, dense, group_colptr, group_col});
    return out[0];
  }

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto has_value = ctx->saved_data["has_value"].toBool();
    auto saved = ctx->get_saved_variables();
    auto rowptr = saved[0], col = saved[1], values = saved[2], dense = saved[3],
         group_colptr = saved[4], group_col = saved[5];

    auto grad_value = torch::Tensor();
    if (has_value > 0 &&
        torch::autograd::any_variable_requires_grad({values})) {
      grad_value = sddmm_cuda_csr(rowptr, col, grad_out, dense);
    }

    auto grad_mat = std::vector<torch::Tensor>();
    if (torch::autograd::any_variable_requires_grad({dense})) {
      auto t_values = torch::Tensor();
      auto colptr = torch::Tensor();
      auto row = torch::Tensor();
      // if (has_value)
      auto ten_vec = csr2csc_cuda(rowptr, col, values);
      colptr = ten_vec[0];
      row = ten_vec[1];
      t_values = ten_vec[2];
      grad_mat = spmm_group_cuda(colptr, row, t_values, grad_out, group_colptr,
                                 group_col, has_value);
    }
    return {torch::Tensor(), torch::Tensor(), grad_value,
            grad_mat[0],     torch::Tensor(), torch::Tensor(),
            torch::Tensor(), torch::Tensor(), torch::Tensor()};
    //       has_value};
  }
};

torch::Tensor spmm_sum_group(torch::Tensor rowptr, torch::Tensor col,
                             torch::Tensor values, torch::Tensor dense,
                             torch::Tensor group_rowptr,
                             torch::Tensor group_row,
                             torch::Tensor group_colptr,
                             torch::Tensor group_col, bool has_value) {
  return SpMMSum::apply(rowptr, col, values, dense, group_rowptr, group_row,
                        group_colptr, group_col, has_value);
}

TORCH_LIBRARY(dgsparse_spmm_group, m) {
  m.def("spmm_sum_group", &spmm_sum_group);
  // m.def("spmm_max", &spmm_max);
  // m.def("spmm_min", &spmm_min);
  // m.def("spmm_mean", &spmm_mean);
  // m.def("csr2csc", &csr2csc);
}
