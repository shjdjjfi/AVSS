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

# Set random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Set offline mode
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Load DistilBERT model and Tokenizer
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add padding token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Ensure using padding token
tokenizer.pad_token = '[PAD]'  # Or use an appropriate token

# Load SST-2 dataset
train_dataset = load_dataset("csv", data_files="/root/sst2/data/train.csv")['train']
val_dataset = load_dataset("csv", data_files="/root/sst2/data/validation.csv")['train']

# Place train and val datasets into DatasetDict
dataset = DatasetDict({
    "train": train_dataset,  
    "val": val_dataset  
})

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])  # Ensure label column is included

# Create DataLoader
val_loader = DataLoader(tokenized_dataset['val'], batch_size=8, shuffle=True)

# Move model to GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define hooks to capture output of each layer
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output[0].detach().cpu()  # Only capture the first output
    return hook

# Register forward hooks to each layer
hooks = []
for i, layer in enumerate(model.distilbert.transformer.layer):
    hooks.append(layer.register_forward_hook(get_activation(f"layer_{i}")))

# Define functions to calculate activation distribution complexity and sparsity
def calculate_activation_variance(layer_output):
    pooled_output = layer_output.mean(dim=1)  # Take the mean across the sequence length dimension
    variance = pooled_output.var(dim=1)  # Calculate variance across the hidden dimension
    avg_variance = variance.mean().item()
    return avg_variance

def calculate_l1_norm(gradients):
    l1_norm = gradients.abs().sum().item()
    return l1_norm

# Calculate gradient L2 norm
def calculate_gradient_l2_norm(gradients):
    l2_norm = gradients.pow(2).sum().sqrt().item()
    return l2_norm

# Calculate Frobenius norm
def calculate_frobenius_norm(weight_matrix):
    frobenius_norm = torch.norm(weight_matrix, p='fro').item()
    return frobenius_norm

# Calculate activation sparsity
def calculate_activation_sparsity(layer_output):
    sparsity = (layer_output.abs() < 1e-5).float().mean().item()  # Proportion of activations close to zero
    return sparsity

# Store metrics for each layer
layer_variances = {i: [] for i in range(len(model.distilbert.transformer.layer))}
layer_l1_norms = {i: [] for i in range(len(model.distilbert.transformer.layer))}
layer_l2_norms = {i: [] for i in range(len(model.distilbert.transformer.layer))}
layer_frobenius_norms = {i: [] for i in range(len(model.distilbert.transformer.layer))}
layer_sparsities = {i: [] for i in range(len(model.distilbert.transformer.layer))}

# Iterate over the data and calculate various metrics
for batch in tqdm(val_loader, desc="Processing batches"):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['label'].to(device)  # Extract labels from the dataset

    # Forward pass
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss

    # Backpropagate to calculate gradients
    loss.backward()

    for i in range(len(model.distilbert.transformer.layer)):
        layer_output = activations[f"layer_{i}"]

        # Calculate activation variance and sparsity
        var = calculate_activation_variance(layer_output)
        sparsity = calculate_activation_sparsity(layer_output)

        # Get gradients from the sub-modules containing weights in the layer
        layer = model.distilbert.transformer.layer[i]

        # Get gradients from the linear layers in the feed-forward network
        linear1_weight_grad = layer.ffn.lin1.weight.grad  # Modified to lin1
        linear2_weight_grad = layer.ffn.lin2.weight.grad  # Modified to lin2
        l1_norm_ffn = calculate_l1_norm(linear1_weight_grad) + calculate_l1_norm(linear2_weight_grad)
        l2_norm_ffn = calculate_gradient_l2_norm(linear1_weight_grad) + calculate_gradient_l2_norm(linear2_weight_grad)
        frobenius_norm_ffn = calculate_frobenius_norm(layer.ffn.lin1.weight) + calculate_frobenius_norm(layer.ffn.lin2.weight)

        # Store results
        layer_variances[i].append(var)
        layer_l1_norms[i].append(l1_norm_ffn)
        layer_l2_norms[i].append(l2_norm_ffn)
        layer_frobenius_norms[i].append(frobenius_norm_ffn)
        layer_sparsities[i].append(sparsity)

    # Clear gradients
    model.zero_grad()

# Remove hooks
for hook in hooks:
    hook.remove()

# Calculate average metrics for each layer
average_variances = {layer: np.mean(vars) for layer, vars in layer_variances.items()}
average_l1_norms = {layer: np.mean(norms) for layer, norms in layer_l1_norms.items()}
average_l2_norms = {layer: np.mean(norms) for layer, norms in layer_l2_norms.items()}
average_frobenius_norms = {layer: np.mean(norms) for layer, norms in layer_frobenius_norms.items()}
average_sparsities = {layer: np.mean(sps) for layer, sps in layer_sparsities.items()}

# Create DataFrame for easier analysis
metric_info = pd.DataFrame({
    'Layer': list(average_variances.keys()),
    'Average_Variance': list(average_variances.values()),
    'Average_L1Norm': list(average_l1_norms.values()),
    'Average_L2Norm': list(average_l2_norms.values()),
    'Average_FrobeniusNorm': list(average_frobenius_norms.values()),
    'Average_Sparsity': list(average_sparsities.values())
})

# Visualize each metric
fig, axs = plt.subplots(3, 2, figsize=(12, 12))

# Plot activation variance
axs[0, 0].bar(metric_info['Layer'], metric_info['Average_Variance'], alpha=0.6)
axs[0, 0].set_title('Activation Variance per Layer')
axs[0, 0].set_xlabel('Layer')
axs[0, 0].set_ylabel('Variance')

# Plot L1 norm
axs[0, 1].bar(metric_info['Layer'], metric_info['Average_L1Norm'], alpha=0.6)
axs[0, 1].set_title('L1 Norm per Layer')
axs[0, 1].set_xlabel('Layer')
axs[0, 1].set_ylabel('L1 Norm')

# Plot L2 norm
axs[1, 0].bar(metric_info['Layer'], metric_info['Average_L2Norm'], alpha=0.6)
axs[1, 0].set_title('L2 Norm per Layer')
axs[1, 0].set_xlabel('Layer')
axs[1, 0].set_ylabel('L2 Norm')

# Plot Frobenius norm
axs[1, 1].bar(metric_info['Layer'], metric_info['Average_FrobeniusNorm'], alpha=0.6)
axs[1, 1].set_title('Frobenius Norm per Layer')
axs[1, 1].set_xlabel('Layer')
axs[1, 1].set_ylabel('Frobenius Norm')

# Plot activation sparsity
axs[2, 0].bar(metric_info['Layer'], metric_info['Average_Sparsity'], alpha=0.6)
axs[2, 0].set_title('Activation Sparsity per Layer')
axs[2, 0].set_xlabel('Layer')
axs[2, 0].set_ylabel('Sparsity')

# Hide empty subplot
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
