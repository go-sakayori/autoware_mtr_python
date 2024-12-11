#include "knn_func.h"

void knn_batch_wrapper(
  at::Tensor xyz_tensor, at::Tensor query_xyz_tensor, at::Tensor batch_idxs_tensor,
  at::Tensor query_batch_offsets_tensor, at::Tensor idx_tensor, int n, int m, int k)
{
  const float * query_xyz = query_xyz_tensor.data_ptr<float>();
  const float * xyz = xyz_tensor.data_ptr<float>();
  const int * batch_idxs = batch_idxs_tensor.data_ptr<int>();
  const int * query_batch_offsets = query_batch_offsets_tensor.data_ptr<int>();
  int * idx = idx_tensor.data_ptr<int>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  knn_batch_cuda(n, m, k, xyz, query_xyz, batch_idxs, query_batch_offsets, idx, stream);
}

void knn_batch_mlogk_wrapper(
  at::Tensor xyz_tensor, at::Tensor query_xyz_tensor, at::Tensor batch_idxs_tensor,
  at::Tensor query_batch_offsets_tensor, at::Tensor idx_tensor, int n, int m, int k)
{
  const float * query_xyz = query_xyz_tensor.data_ptr<float>();
  const float * xyz = xyz_tensor.data_ptr<float>();
  const int * batch_idxs = batch_idxs_tensor.data_ptr<int>();
  const int * query_batch_offsets = query_batch_offsets_tensor.data_ptr<int>();
  int * idx = idx_tensor.data_ptr<int>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  knn_batch_mlogk_cuda(n, m, k, xyz, query_xyz, batch_idxs, query_batch_offsets, idx, stream);
}
