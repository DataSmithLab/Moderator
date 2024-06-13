import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np

def quantize_per_tensor(weight, num_bits=8):
    qmin = -1.0 * (2**num_bits) / 2
    qmax = -1.0 * qmin - 1
    min_val, max_val = weight.min(), weight.max()

    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - min_val / scale

    # 在这里，我们使用了numpy的round函数来实现四舍五入
    quantized_weight = torch.round(weight / scale + zero_point)

    # 我们需要确保量化后的权重在有效范围内
    quantized_weight = torch.clamp(quantized_weight, qmin, qmax).to(torch.int8)

    return quantized_weight, scale, zero_point

def dequantize_per_tensor(quantized_weight, scale, zero_point):
    return scale * (quantized_weight - zero_point)

def quantize_task_vector(tv_state_dict):
    new_tv_state_dict = {}
    scale_dict = {}
    zero_point_dict = {}
    for key, weight in tv_state_dict.items():
        quantized_weight, scale, zero_point = quantize_per_tensor(weight)
        new_tv_state_dict[key]=quantized_weight
        scale_dict[key]=scale
        zero_point_dict[key]=zero_point
    return new_tv_state_dict, scale_dict, zero_point_dict

def dequantize_task_vector(quantized_state_dict, scale_dict, zero_point_dict):
    original_state_dict = {}
    for key, weight in quantized_state_dict.items():
        scale = scale_dict[key]
        zero_point = zero_point_dict[key]
        original_weight = dequantize_per_tensor(weight, scale, zero_point)
        original_state_dict[key]=original_weight
    return original_state_dict

def prune_weights_ratio(state_dict, prune_ratio=0.1):
    # 遍历模型中的每一层
    for layer_name, layer_weights in state_dict.items():
        # 将权重转换为numpy数组
        weights = layer_weights.cpu().numpy()
        # 计算剪枝的阈值
        threshold = np.percentile(np.abs(weights), prune_ratio * 100)
        # 将小于阈值的权重设为零
        weights[np.abs(weights) < threshold] = 0
        # 更新权重
        state_dict[layer_name] = torch.from_numpy(weights)
    return state_dict

def prune_weights_threshold(state_dict, prune_threshold=1e-4):
    # 遍历模型中的每一层
    for layer_name, layer_weights in state_dict.items():
        # 将权重转换为numpy数组
        weights = layer_weights.cpu().numpy()
        # 计算剪枝的阈值
        threshold = prune_threshold
        # 将小于阈值的权重设为零
        weights[np.abs(weights) < threshold] = 0
        # 更新权重
        state_dict[layer_name] = torch.from_numpy(weights)
    return state_dict