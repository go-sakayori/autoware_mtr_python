#include "attention/attention_func.h"
#include "knn/knn_func.h"

#include <torch/torch.h>

/**
 *
 * REFERENCES:
 * -
 * https://github.com/pytorch/pytorch/blob/2e2cfa2ef77ff95741f7ebc21a24846567d8630b/torch/csrc/autograd/custom_function.h
 * -
 * https://discuss.pytorch.org/t/about-return-none-in-torch-autograd-functions-backward-in-c-api-libtorch/147615/3
 *
 */

struct AttentionWeightComputation : public torch::autograd::Function<AttentionWeightComputation>
{
  static torch::Tensor forward(
    torch::autograd::AutogradContext * ctx, torch::Tensor query_batch_cnt,
    torch::Tensor key_batch_cnt, torch::Tensor index_pair_batch, torch::Tensor index_pair,
    torch::Tensor query_features, torch::Tensor key_features)
  {
    int b = query_batch_cnt.sizes().at(0);

    const auto & query_sizes = index_pair.sizes();
    const auto & key_sizes = key_features.sizes();

    int total_query_num = query_sizes.at(0);
    int local_size = query_sizes.at(1);

    int total_key_num = key_sizes.at(0);
    int nhead = key_sizes.at(1);
    int hdim = key_sizes.at(2);

    // save context
    // non-variable data
    ctx->saved_data["b"] = b;
    ctx->saved_data["total_query_num"] = total_query_num;
    ctx->saved_data["local_size"] = local_size;
    ctx->saved_data["total_key_num"] = total_key_num;
    ctx->saved_data["nhead"] = nhead;
    ctx->saved_data["hdim"] = hdim;
    // variable data
    torch::autograd::variable_list to_save = {query_batch_cnt, key_batch_cnt,  index_pair_batch,
                                              index_pair,      query_features, key_features};
    ctx->save_for_backward(to_save);

    torch::Tensor output =
      torch::zeros({total_query_num, local_size, nhead}, torch::device(torch::kCUDA));

    attention_weight_computation_wrapper(
      b, total_query_num, local_size, total_key_num, nhead, hdim, query_batch_cnt, key_batch_cnt,
      index_pair_batch, index_pair, query_features, key_features, output);

    return output;
  }

  /**
   * @brief Backward function implementation.
   */
  static torch::autograd::variable_list backward(
    torch::autograd::AutogradContext * ctx, torch::autograd::variable_list grad_outputs)
  {
    // load saved context
    auto b = ctx->saved_data["b"].toInt();
    auto total_query_num = ctx->saved_data["total_query_num"].toInt();
    auto local_size = ctx->saved_data["local_size"].toInt();
    auto total_key_num = ctx->saved_data["total_key_num"].toInt();
    auto nhead = ctx->saved_data["nhead"].toInt();
    auto hdim = ctx->saved_data["hdim"].toInt();

    auto saved_variables = ctx->get_saved_variables();

    auto query_batch_cnt = saved_variables.at(0);
    auto key_batch_cnt = saved_variables.at(1);
    auto index_pair_batch = saved_variables.at(2);
    auto index_pair = saved_variables.at(3);
    auto query_features = saved_variables.at(4);
    auto key_features = saved_variables.at(5);

    auto grad_out = grad_outputs.at(0).contiguous();

    auto grad_query_features =
      torch::zeros({total_query_num, nhead, hdim}, torch::device(torch::kCUDA));
    auto grad_key_features =
      torch::zeros({total_key_num, nhead, hdim}, torch::device(torch::kCUDA));

    attention_weight_computation_grad_wrapper(
      b, total_query_num, local_size, total_key_num, nhead, hdim, query_batch_cnt, key_batch_cnt,
      index_pair_batch, index_pair, query_features, key_features, grad_out, grad_query_features,
      grad_key_features);

    torch::Tensor none;
    return {none, none, none, none, grad_query_features, grad_key_features};
  }

};  // struct AttentionWeightComputation

struct AttentionValueComputation : public torch::autograd::Function<AttentionValueComputation>
{
  static torch::Tensor forward(
    torch::autograd::AutogradContext * ctx, torch::Tensor query_batch_cnt,
    torch::Tensor key_batch_cnt, torch::Tensor index_pair_batch, torch::Tensor index_pair,
    torch::Tensor attn_weight, torch::Tensor value_features)
  {
    int b = query_batch_cnt.sizes().at(0);

    const auto & query_sizes = index_pair.sizes();
    const auto & value_sizes = value_features.sizes();

    int total_query_num = query_sizes.at(0);
    int local_size = query_sizes.at(1);

    int total_key_num = value_sizes.at(0);
    int nhead = value_sizes.at(1);
    int hdim = value_sizes.at(2);

    // save context
    // non-variable data
    ctx->saved_data["b"] = b;
    ctx->saved_data["total_query_num"] = total_query_num;
    ctx->saved_data["local_size"] = local_size;
    ctx->saved_data["total_key_num"] = total_key_num;
    ctx->saved_data["nhead"] = nhead;
    ctx->saved_data["hdim"] = hdim;
    // variable data
    torch::autograd::variable_list to_save = {query_batch_cnt, key_batch_cnt, index_pair_batch,
                                              index_pair,      attn_weight,   value_features};
    ctx->save_for_backward(to_save);

    torch::Tensor output =
      torch::zeros({total_query_num, nhead, hdim}, torch::device(torch::kCUDA));

    attention_value_computation_wrapper(
      b, total_query_num, local_size, total_key_num, nhead, hdim, query_batch_cnt, key_batch_cnt,
      index_pair_batch, index_pair, attn_weight, value_features, output);

    return output;
  }

  static torch::autograd::variable_list backward(
    torch::autograd::AutogradContext * ctx, torch::autograd::variable_list grad_outputs)
  {
    // load saved context
    auto b = ctx->saved_data["b"].toInt();
    auto total_query_num = ctx->saved_data["total_query_num"].toInt();
    auto local_size = ctx->saved_data["local_size"].toInt();
    auto total_key_num = ctx->saved_data["total_key_num"].toInt();
    auto nhead = ctx->saved_data["nhead"].toInt();
    auto hdim = ctx->saved_data["hdim"].toInt();

    auto saved_variables = ctx->get_saved_variables();

    auto query_batch_cnt = saved_variables.at(0);
    auto key_batch_cnt = saved_variables.at(1);
    auto index_pair_batch = saved_variables.at(2);
    auto index_pair = saved_variables.at(3);
    auto attn_weight = saved_variables.at(4);
    auto value_features = saved_variables.at(5);

    auto grad_out = grad_outputs.at(0).contiguous();

    auto grad_attn_weight =
      torch::zeros({total_query_num, local_size, nhead}, torch::device(torch::kCUDA));
    auto grad_value_features =
      torch::zeros({total_key_num, nhead, hdim}, torch::device(torch::kCUDA));

    attention_value_computation_grad_wrapper(
      b, total_query_num, local_size, total_key_num, nhead, hdim, query_batch_cnt, key_batch_cnt,
      index_pair_batch, index_pair, attn_weight, value_features, grad_out, grad_attn_weight,
      grad_value_features);

    torch::Tensor none;
    return {none, none, none, none, grad_attn_weight, grad_value_features};
  }
};  // struct AttentionValueComputation

torch::Tensor attention_weight_computation(
  torch::Tensor query_batch_cnt, torch::Tensor key_batch_cnt, torch::Tensor index_pair_batch,
  torch::Tensor index_pair, torch::Tensor query_features, torch::Tensor key_features)
{
  return AttentionWeightComputation::apply(
    query_batch_cnt, key_batch_cnt, index_pair_batch, index_pair, query_features, key_features);
}

torch::Tensor attention_value_computation(
  torch::Tensor query_batch_cnt, torch::Tensor key_batch_cnt, torch::Tensor index_pair_batch,
  torch::Tensor index_pair, torch::Tensor attn_weight, torch::Tensor value_features)
{
  return AttentionValueComputation::apply(
    query_batch_cnt, key_batch_cnt, index_pair_batch, index_pair, attn_weight, value_features);
}

struct KNNBatch : public torch::autograd::Function<KNNBatch>
{
  static torch::Tensor forward(
    torch::autograd::AutogradContext * ctx, torch::Tensor xyz, torch::Tensor query_xyz,
    torch::Tensor batch_idxs, torch::Tensor query_batch_offsets, int64_t topk)
  {
    int n = xyz.sizes().at(0);
    int m = query_xyz.sizes().at(0);
    TORCH_CHECK(topk <= m);
    TORCH_CHECK(xyz.is_contiguous() && xyz.device().type() == torch::kCUDA);
    TORCH_CHECK(query_xyz.is_contiguous() && query_xyz.device().type() == torch::kCUDA);
    TORCH_CHECK(batch_idxs.is_contiguous() && batch_idxs.device().type() == torch::kCUDA);
    TORCH_CHECK(
      query_batch_offsets.is_contiguous() && query_batch_offsets.device().type() == torch::kCUDA);

    torch::Tensor idx = torch::zeros({n, topk}, torch::kInt).to(torch::kCUDA);

    knn_batch_wrapper(xyz, query_xyz, batch_idxs, query_batch_offsets, idx, n, m, topk);

    return idx;
  }

  static torch::autograd::variable_list backward(
    torch::autograd::AutogradContext * _ctx, torch::autograd::variable_list _grad)
  {
    torch::Tensor none;
    return {none, none, none, none, none};
  }
};  // struct KNNBatch

struct KNNBatchMlogK : public torch::autograd::Function<KNNBatchMlogK>
{
  static torch::Tensor forward(
    torch::autograd::AutogradContext * ctx, torch::Tensor xyz, torch::Tensor query_xyz,
    torch::Tensor batch_idxs, torch::Tensor query_batch_offsets, int64_t topk)
  {
    int n = xyz.sizes().at(0);
    int m = query_xyz.sizes().at(0);
    TORCH_CHECK(xyz.is_contiguous() && xyz.device().type() == torch::kCUDA);
    TORCH_CHECK(query_xyz.is_contiguous() && query_xyz.device().type() == torch::kCUDA);
    TORCH_CHECK(batch_idxs.is_contiguous() && batch_idxs.device().type() == torch::kCUDA);
    TORCH_CHECK(
      query_batch_offsets.is_contiguous() && query_batch_offsets.device().type() == torch::kCUDA);
    TORCH_CHECK(topk <= 128);

    torch::Tensor idx = torch::zeros({n, topk}, torch::kInt).to(torch::kCUDA);

    knn_batch_mlogk_wrapper(xyz, query_xyz, batch_idxs, query_batch_offsets, idx, n, m, topk);

    return idx;
  }

  static torch::autograd::variable_list backward(
    torch::autograd::AutogradContext * _ctx, torch::autograd::variable_list _grad)
  {
    torch::Tensor none;
    return {none, none, none, none, none};
  }
};  // struct KNNBatchMlogK

torch::Tensor knn_batch(
  torch::Tensor xyz, torch::Tensor query_xyz, torch::Tensor batch_idxs,
  torch::Tensor query_batch_offsets, int64_t topk)
{
  return KNNBatch::apply(xyz, query_xyz, batch_idxs, query_batch_offsets, topk);
}

torch::Tensor knn_batch_mlogk(
  torch::Tensor xyz, torch::Tensor query_xyz, torch::Tensor batch_idxs,
  torch::Tensor query_batch_offsets, int64_t topk)
{
  return KNNBatchMlogK::apply(xyz, query_xyz, batch_idxs, query_batch_offsets, topk);
}

TORCH_LIBRARY(awml_pred, m)
{
  m.def(
     "attention_weight_computation(Tensor query_batch_cnt, Tensor key_batch_cnt, Tensor "
     "index_pair_batch, Tensor index_pair, Tensor query_features, Tensor key_features) -> Tensor")
    .def(
      "attention_value_computation(Tensor query_batch_cnt, Tensor key_batch_cnt, Tensor "
      "index_pair_batch, Tensor index_pair, Tensor attn_weight, Tensor value_features) -> Tensor")
    .def(
      "knn_batch(Tensor xyz, Tensor query_xyz, Tensor batch_idxs, Tensor query_batch_offsets, "
      "int topk) -> Tensor")
    .def(
      "knn_batch_mlogk(Tensor xyz, Tensor query_xyz, Tensor batch_idxs, Tensor "
      "query_batch_offsets, "
      "int topk) -> Tensor");
}

TORCH_LIBRARY_IMPL(awml_pred, Autograd, m)
{
  m.impl("attention_weight_computation", &attention_weight_computation)
    .impl("attention_value_computation", &attention_value_computation)
    .impl("knn_batch", &knn_batch)
    .impl("knn_batch_mlogk", &knn_batch_mlogk);
}
