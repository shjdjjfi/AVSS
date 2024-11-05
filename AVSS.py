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
