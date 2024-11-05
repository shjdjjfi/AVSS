import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import random
import os
import pandas as pd
import matplotlib.pyplot as plt

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 设置离线模式
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 加载 StableLM 模型和 Tokenizer
model_name = "stablelm-base-alpha-3b-v2"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 加载 SQuAD 数据集并取前 10% 的数据
dataset = load_dataset("parquet", data_files={
    "train": "squad/plain_text/train-00000-of-00001.parquet",
    "validation": "squad/plain_text/validation-00000-of-00001.parquet"
})
def get_subset(dataset, percentage=0.01):
    num_samples = int(len(dataset) * percentage)
    return dataset.select(range(num_samples))
dataset['train'] = get_subset(dataset['train'])
dataset['validation'] = get_subset(dataset['validation'])

# 数据预处理
def preprocess_function(examples):
    inputs = ["问题：" + q + " 上下文：" + c for q, c in zip(examples["question"], examples["context"])]
    targets = [a["text"][0] if len(a["text"]) > 0 else "" for a in examples["answers"]]
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=256)
    labels = tokenizer(targets, padding="max_length", truncation=True, max_length=256).input_ids
    labels = [[-100 if token == tokenizer.pad_token_id else token for token in label] for label in labels]
    model_inputs["labels"] = labels
    return model_inputs

# 预处理并设置格式
tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_loader = DataLoader(tokenized_dataset['validation'], batch_size=8, shuffle=True)

# 将模型移动到 GPU 或 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 钩子函数来捕获每层的输出
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output[0].detach().cpu()
    return hook

# 注册 forward hook 到每层
hooks = []
for i, layer in enumerate(model.transformer.layers):
    hooks.append(layer.register_forward_hook(get_activation(f"layer_{i}")))

# 定义度量计算函数
def calculate_activation_variance(layer_output):
    pooled_output = layer_output.mean(dim=1)
    variance = pooled_output.var(dim=1)
    avg_variance = variance.mean().item()
    return avg_variance

def calculate_l1_norm(gradients):
    l1_norm = gradients.abs().sum().item()
    return l1_norm

def calculate_gradient_l2_norm(gradients):
    l2_norm = gradients.pow(2).sum().sqrt().item()
    return l2_norm

def calculate_frobenius_norm(weight_matrix):
    frobenius_norm = torch.norm(weight_matrix, p='fro').item()
    return frobenius_norm

def calculate_activation_sparsity(layer_output):
    sparsity = (layer_output.abs() < 1e-5).float().mean().item()
    return sparsity

# 存储每层的度量
layer_variances = {i: [] for i in range(len(model.transformer.layers))}
layer_l1_norms = {i: [] for i in range(len(model.transformer.layers))}
layer_l2_norms = {i: [] for i in range(len(model.transformer.layers))}
layer_frobenius_norms = {i: [] for i in range(len(model.transformer.layers))}
layer_sparsities = {i: [] for i in range(len(model.transformer.layers))}

# 遍历数据并计算度量
for batch in tqdm(val_loader, desc="Processing batches"):
    input_ids = batch['input_ids'].to(device)

    # 前向传播并计算损失
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss

    # 反向传播以计算梯度
    loss.backward()

    for i in range(len(model.transformer.layers)):
        layer_output = activations[f"layer_{i}"]
        var = calculate_activation_variance(layer_output)
        sparsity = calculate_activation_sparsity(layer_output)

        # 获取注意力和前馈层的梯度和权重矩阵
        layer = model.transformer.layers[i]
        attn_weight_grad = layer.attention.qkv_proj.weight.grad
        linear1_weight_grad = layer.mlp.gate_proj.weight.grad
        linear2_weight_grad = layer.mlp.out_proj.weight.grad
        total_l1_norm = calculate_l1_norm(attn_weight_grad) + calculate_l1_norm(linear1_weight_grad) + calculate_l1_norm(linear2_weight_grad)
        total_l2_norm = calculate_gradient_l2_norm(attn_weight_grad) + calculate_gradient_l2_norm(linear1_weight_grad) + calculate_gradient_l2_norm(linear2_weight_grad)
        total_frobenius_norm = calculate_frobenius_norm(layer.attention.qkv_proj.weight) + calculate_frobenius_norm(layer.mlp.gate_proj.weight) + calculate_frobenius_norm(layer.mlp.out_proj.weight)

        layer_variances[i].append(var)
        layer_l1_norms[i].append(total_l1_norm)
        layer_l2_norms[i].append(total_l2_norm)
        layer_frobenius_norms[i].append(total_frobenius_norm)
        layer_sparsities[i].append(sparsity)

    # 清空梯度
    model.zero_grad()
    
# 取消 hooks 注册
for hook in hooks:
    hook.remove()

# 计算每层的平均度量
average_variances = {layer: np.mean(vars) for layer, vars in layer_variances.items()}
average_l1_norms = {layer: np.mean(norms) for layer, norms in layer_l1_norms.items()}
average_l2_norms = {layer: np.mean(norms) for layer, norms in layer_l2_norms.items()}
average_frobenius_norms = {layer: np.mean(norms) for layer, norms in layer_frobenius_norms.items()}
average_sparsities = {layer: np.mean(sps) for layer, sps in layer_sparsities.items()}

# 创建 DataFrame 以便于分析
metric_info = pd.DataFrame({
    'Layer': list(average_variances.keys()),
    'Average_Variance': list(average_variances.values()),
    'Average_L1Norm': list(average_l1_norms.values()),
    'Average_L2Norm': list(average_l2_norms.values()),
    'Average_FrobeniusNorm': list(average_frobenius_norms.values()),
    'Average_Sparsity': list(average_sparsities.values())
})

# 可视化每种度量方法
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# 激活方差
axs[0, 0].bar(metric_info['Layer'], metric_info['Average_Variance'], alpha=0.7)
axs[0, 0].set_title('Activation Variance per Layer')
axs[0, 0].set_xlabel('Layer')
axs[0, 0].set_ylabel('Variance')

# L1范数
axs[0, 1].bar(metric_info['Layer'], metric_info['Average_L1Norm'], alpha=0.7)
axs[0, 1].set_title('L1 Norm per Layer')
axs[0, 1].set_xlabel('Layer')
axs[0, 1].set_ylabel('L1 Norm')

# L2范数
axs[0, 2].bar(metric_info['Layer'], metric_info['Average_L2Norm'], alpha=0.7)
axs[0, 2].set_title('L2 Norm per Layer')
axs[0, 2].set_xlabel('Layer')
axs[0, 2].set_ylabel('L2 Norm')

# Frobenius范数
axs[1, 0].bar(metric_info['Layer'], metric_info['Average_FrobeniusNorm'], alpha=0.7)
axs[1, 0].set_title('Frobenius Norm per Layer')
axs[1, 0].set_xlabel('Layer')
axs[1, 0].set_ylabel('Frobenius Norm')

# 激活稀疏性
axs[1, 1].bar(metric_info['Layer'], metric_info['Average_Sparsity'], alpha=0.7)
axs[1, 1].set_title('Activation Sparsity per Layer')
axs[1, 1].set_xlabel('Layer')
axs[1, 1].set_ylabel('Sparsity')

# 隐藏空白子图
axs[1, 2].axis('off')

plt.tight_layout()
plt.show()