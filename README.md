# AVSS: Layer Importance Evaluation in Large Language Models via Activation Variance-Sparsity Analysis

## Project Overview

This repository contains the official implementation of the paper **"AVSS: Layer Importance Evaluation in Large Language Models via Activation Variance-Sparsity Analysis"**. The evaluation of layer importance in deep learning has been an active area of research, with significant implications for model optimization and interpretability. Recently, large language models (LLMs) have gained prominence across various domains, yet limited studies have explored the functional importance and performance contributions of individual layers within LLMs, especially from the perspective of activation distribution. In this work, we propose the **Activation Variance-Sparsity Score (AVSS)**, a novel metric that combines normalized activation variance and sparsity to assess each layer’s contribution to model performance. By identifying and removing approximately the lowest 25% of layers based on AVSS, we retain over 90% of the original model's performance across tasks such as question answering, language modeling, and sentiment classification, suggesting that these layers may be non-essential. Our approach provides a systematic method for identifying less critical layers, contributing to more efficient large language model architectures.

## Paper Link
```bibtex
@misc{song2024avss,
    title={AVSS: Layer Importance Evaluation in Large Language Models via Activation Variance-Sparsity Analysis},
    author={Zichen Song and Yuxin Wu and Sitan Huang and Zhongfeng Kang},
    year={2024},
    eprint={2411.02117},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## The AVSS method
The AVSS method consists of two main steps: (1) Evaluating the activation sparsity and activation distribution variance for all layers in the large language model; (2) Using the AVSS to calculate the importance of each layer. Here, I will use the SST2 dataset and the DistilBERT model as an example for demonstration.

### Evaluating the activation sparsity and activation distribution variance
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset, DatasetDict
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

# 加载 DistilBERT 模型和 Tokenizer
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 添加填充标记
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 确保使用填充标记
tokenizer.pad_token = '[PAD]'  # 或者使用适合的标记

# 加载 SST-2 数据集
train_dataset = load_dataset("csv", data_files="/root/sst2/data/train.csv")['train']
val_dataset = load_dataset("csv", data_files="/root/sst2/data/validation.csv")['train']

# 将 train 和 val 数据集放入 DatasetDict 中
dataset = DatasetDict({
    "train": train_dataset,  
    "val": val_dataset  
})

# 对数据进行tokenization
def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])  # 确保包含 label 列

# 创建 DataLoader
val_loader = DataLoader(tokenized_dataset['val'], batch_size=8, shuffle=True)

# 将模型移动到 GPU 或 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义钩子函数来捕获每一层的输出
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output[0].detach().cpu()  # 只获取第一个输出
    return hook

# 注册 forward hook 到每一层
hooks = []
for i, layer in enumerate(model.distilbert.transformer.layer):
    hooks.append(layer.register_forward_hook(get_activation(f"layer_{i}")))

# 定义激活分布复杂度和稀疏性计算函数
def calculate_activation_variance(layer_output):
    pooled_output = layer_output.mean(dim=1)  # 在序列长度维度上取平均
    variance = pooled_output.var(dim=1)  # 计算隐藏维度上的方差
    avg_variance = variance.mean().item()
    return avg_variance

def calculate_l1_norm(gradients):
    l1_norm = gradients.abs().sum().item()
    return l1_norm

# 计算梯度L2范数
def calculate_gradient_l2_norm(gradients):
    l2_norm = gradients.pow(2).sum().sqrt().item()
    return l2_norm

# 计算Frobenius范数
def calculate_frobenius_norm(weight_matrix):
    frobenius_norm = torch.norm(weight_matrix, p='fro').item()
    return frobenius_norm

# 计算激活稀疏性
def calculate_activation_sparsity(layer_output):
    sparsity = (layer_output.abs() < 1e-5).float().mean().item()  # 激活接近零的比例
    return sparsity

# 存储每一层的度量值
layer_variances = {i: [] for i in range(len(model.distilbert.transformer.layer))}
layer_l1_norms = {i: [] for i in range(len(model.distilbert.transformer.layer))}
layer_l2_norms = {i: [] for i in range(len(model.distilbert.transformer.layer))}
layer_frobenius_norms = {i: [] for i in range(len(model.distilbert.transformer.layer))}
layer_sparsities = {i: [] for i in range(len(model.distilbert.transformer.layer))}

# 遍历数据并计算各种度量
for batch in tqdm(val_loader, desc="Processing batches"):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['label'].to(device)  # 从数据集中提取标签

    # 前向传播
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss

    # 反向传播来计算梯度
    loss.backward()

    for i in range(len(model.distilbert.transformer.layer)):
        layer_output = activations[f"layer_{i}"]

        # 计算激活方差和稀疏性
        var = calculate_activation_variance(layer_output)
        sparsity = calculate_activation_sparsity(layer_output)

        # 获取对应层中包含权重的子模块的梯度
        layer = model.distilbert.transformer.layer[i]

        # 获取前馈网络中的线性层的梯度
        linear1_weight_grad = layer.ffn.lin1.weight.grad  # 修改为 lin1
        linear2_weight_grad = layer.ffn.lin2.weight.grad  # 修改为 lin2
        l1_norm_ffn = calculate_l1_norm(linear1_weight_grad) + calculate_l1_norm(linear2_weight_grad)
        l2_norm_ffn = calculate_gradient_l2_norm(linear1_weight_grad) + calculate_gradient_l2_norm(linear2_weight_grad)
        frobenius_norm_ffn = calculate_frobenius_norm(layer.ffn.lin1.weight) + calculate_frobenius_norm(layer.ffn.lin2.weight)

        # 存储结果
        layer_variances[i].append(var)
        layer_l1_norms[i].append(l1_norm_ffn)
        layer_l2_norms[i].append(l2_norm_ffn)
        layer_frobenius_norms[i].append(frobenius_norm_ffn)
        layer_sparsities[i].append(sparsity)

    # 清空梯度
    model.zero_grad()


# 取消注册 hooks
for hook in hooks:
    hook.remove()

# 计算每一层的平均度量值
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
fig, axs = plt.subplots(3, 2, figsize=(12, 12))

# 绘制激活方差
axs[0, 0].bar(metric_info['Layer'], metric_info['Average_Variance'], alpha=0.6)
axs[0, 0].set_title('Activation Variance per Layer')
axs[0, 0].set_xlabel('Layer')
axs[0, 0].set_ylabel('Variance')

# 绘制L1范数
axs[0, 1].bar(metric_info['Layer'], metric_info['Average_L1Norm'], alpha=0.6)
axs[0, 1].set_title('L1 Norm per Layer')
axs[0, 1].set_xlabel('Layer')
axs[0, 1].set_ylabel('L1 Norm')

# 绘制L2范数
axs[1, 0].bar(metric_info['Layer'], metric_info['Average_L2Norm'], alpha=0.6)
axs[1, 0].set_title('L2 Norm per Layer')
axs[1, 0].set_xlabel('Layer')
axs[1, 0].set_ylabel('L2 Norm')

# 绘制Frobenius范数
axs[1, 1].bar(metric_info['Layer'], metric_info['Average_FrobeniusNorm'], alpha=0.6)
axs[1, 1].set_title('Frobenius Norm per Layer')
axs[1, 1].set_xlabel('Layer')
axs[1, 1].set_ylabel('Frobenius Norm')

# 绘制激活稀疏性
axs[2, 0].bar(metric_info['Layer'], metric_info['Average_Sparsity'], alpha=0.6)
axs[2, 0].set_title('Activation Sparsity per Layer')
axs[2, 0].set_xlabel('Layer')
axs[2, 0].set_ylabel('Sparsity')

# 隐藏空白子图
axs[2, 1].axis('off')

plt.tight_layout()
plt.show()
```

### Computing AVSS
```python
import pandas as pd

# Load activation variance and sparsity data
variance_df = pd.read_csv('a.csv')
sparsity_df = pd.read_csv('b.csv')

# Ensure the files contain a single column with variance and sparsity data, respectively
# Rename columns for clarity if needed
variance_df.columns = ['Layer', 'Variance']
sparsity_df.columns = ['Layer', 'Sparsity']

# Merge the two dataframes on the 'Layer' column
data = pd.merge(variance_df, sparsity_df, on='Layer')

# Calculate the AVSS for each layer
data['AVSS'] = data['Variance'] / data['Sparsity']

# Normalize AVSS by dividing by the sum of all AVSS scores
data['Normalized_AVSS'] = data['AVSS'] / data['AVSS'].sum()

# Calculate cumulative AVSS for pruning decision
data['Cumulative_AVSS'] = data['Normalized_AVSS'].cumsum()

# Sort layers by AVSS score for ranking and pruning decision
data = data.sort_values(by='AVSS', ascending=True)

# Identify layers that contribute minimally and may be pruned (e.g., lowest 25% based on cumulative AVSS)
threshold = 0.25
layers_to_prune = data[data['Cumulative_AVSS'] <= threshold]

# Display layers recommended for pruning
print("Layers recommended for pruning:\n", layers_to_prune)
```
