import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
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

# 加载 StableLM 模型和 Tokenizer
model_name = "stablelm-base-alpha-3b-v2"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 设置 pad_token
tokenizer.pad_token = tokenizer.eos_token

# 加载 val 和 test 数据集
val_dataset = load_dataset("json", data_files="HackerNews/val.json")['train']
test_dataset = load_dataset("json", data_files="HackerNews/test.json")['train']

# 只保留前5%
val_dataset = val_dataset.select(range(int(0.05 * len(val_dataset))))
test_dataset = test_dataset.select(range(int(0.05 * len(test_dataset))))

# 将 val 和 test 数据集放入 DatasetDict 中
dataset = DatasetDict({
    "val": val_dataset,  
    "test": test_dataset  
})

# 对数据进行tokenization
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids"])

# 创建 DataLoader
val_loader = DataLoader(tokenized_dataset['val'], batch_size=8, shuffle=True)

# 将模型移动到 GPU 或 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义钩子函数来捕获每一层的输出
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output[0].detach().cpu()
    return hook

# 注册 forward hook 到每一层
hooks = []
for i, layer in enumerate(model.transformer.layers):
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
layer_variances = {i: [] for i in range(len(model.transformer.layers))}
layer_l1_norms = {i: [] for i in range(len(model.transformer.layers))}
layer_l2_norms = {i: [] for i in range(len(model.transformer.layers))}
layer_frobenius_norms = {i: [] for i in range(len(model.transformer.layers))}
layer_sparsities = {i: [] for i in range(len(model.transformer.layers))}

# 遍历数据并计算各种度量
for batch in tqdm(val_loader, desc="Processing batches"):
    input_ids = batch['input_ids'].to(device)
    labels = input_ids  

    # 前向传播
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss

    # 反向传播来计算梯度
    loss.backward()

    for i in range(len(model.transformer.layers)):
        layer_output = activations[f"layer_{i}"]

        # 计算激活方差和稀疏性
        var = calculate_activation_variance(layer_output)
        sparsity = calculate_activation_sparsity(layer_output)

        # 获取对应层中包含权重的子模块的梯度
        layer = model.transformer.layers[i]

        # 获取注意力机制中的权重（qkv_proj）
        attn_weight_grad = layer.attention.qkv_proj.weight.grad
        l1_norm_attn = calculate_l1_norm(attn_weight_grad)
        l2_norm_attn = calculate_gradient_l2_norm(attn_weight_grad)
        frobenius_norm_attn = calculate_frobenius_norm(layer.attention.qkv_proj.weight)

        # 获取前馈网络中的线性层的梯度
        linear1_weight_grad = layer.mlp.gate_proj.weight.grad
        linear2_weight_grad = layer.mlp.out_proj.weight.grad
        l1_norm_ffn = calculate_l1_norm(linear1_weight_grad) + calculate_l1_norm(linear2_weight_grad)
        l2_norm_ffn = calculate_gradient_l2_norm(linear1_weight_grad) + calculate_gradient_l2_norm(linear2_weight_grad)
        frobenius_norm_ffn = calculate_frobenius_norm(layer.mlp.gate_proj.weight) + calculate_frobenius_norm(layer.mlp.out_proj.weight)

        # 将所有部分的度量值加起来
        total_l1_norm = l1_norm_attn + l1_norm_ffn
        total_l2_norm = l2_norm_attn + l2_norm_ffn
        total_frobenius_norm = frobenius_norm_attn + frobenius_norm_ffn

        # 存储结果
        layer_variances[i].append(var)
        layer_l1_norms[i].append(total_l1_norm)
        layer_l2_norms[i].append(total_l2_norm)
        layer_frobenius_norms[i].append(total_frobenius_norm)
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