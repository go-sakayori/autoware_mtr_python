import os.path as osp

import torch

torch.ops.load_library(osp.join(osp.dirname(__file__), "cuda_ops.so"))

# attention
attention_weight_computation = torch.ops.awml_pred.attention_weight_computation
attention_value_computation = torch.ops.awml_pred.attention_value_computation
# knn
knn_batch = torch.ops.awml_pred.knn_batch
knn_batch_mlogk = torch.ops.awml_pred.knn_batch_mlogk
