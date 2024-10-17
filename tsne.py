import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem, DataStructs
from rdkit.Chem import rdchem, Descriptors, rdMolDescriptors, AllChem, rdFingerprintGenerator
from rdkit.Chem import rdFingerprintGenerator as fpg

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Convert ECFP_BitVect to a numpy array
ecfp_data = np.array(result_df['ECFP_BitVect'].tolist())

scaler = StandardScaler()
standardized_ecfp_data = scaler.fit_transform(ecfp_data)

# t-SNE Transformation
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
tsne_results = tsne.fit_transform(standardized_ecfp_data)

# Create a DataFrame with t-SNE results
tsne_df = pd.DataFrame(data=tsne_results, columns=['t-SNE 1', 't-SNE 2'])
tsne_df['DILI_Classification'] = result_df['DILI_Classification'].values  # 0 for negative, 1 for positive
tsne_df['Source'] = result_df['Source'].str.strip().str.capitalize()  # Normalize 'Source' to 'New' and 'Existing'

# Set up the seaborn style for better visuals
sns.set(style="whitegrid")

# Create a scatter plot with seaborn
plt.figure(figsize=(10, 8))

# Convert DILI classification from numeric to string with descriptive labels
tsne_df['DILI_Label'] = tsne_df['DILI_Classification'].map({0.0: 'Inferred DILI negative', 1.0: 'Inferred DILI positive'})

# Plotting with updated labels and adjusted palette
sns.scatterplot(
    x='t-SNE 1', y='t-SNE 2', 
    hue='DILI_Label',  # Use the new DILI_Label column for coloring
    data=tsne_df[tsne_df['Source'] == 'Existing'],  # Plot only existing molecules
    palette={'Inferred DILI negative': 'lightblue', 'Inferred DILI positive': 'lightcoral'},  # Use descriptive labels as keys
    alpha=0.8,  # Transparency
    s=20  # Small point size
)

# Plot new molecules that are DILI positive (1) in green
sns.scatterplot(
    x='t-SNE 1', y='t-SNE 2',
    data=tsne_df[(tsne_df['Source'] == 'New') & (tsne_df['DILI_Classification'] == 1)],  # New DILI positive molecules
    color='red',  # Green color for DILI positive (1)
    alpha=1,  # No transparency for new molecules
    s=30,  # Slightly larger point size for new molecules
    label='DILIst Positive'  # Add label for legend
)

# Plot new molecules that are DILI negative (0) in purple
sns.scatterplot(
    x='t-SNE 1', y='t-SNE 2',
    data=tsne_df[(tsne_df['Source'] == 'New') & (tsne_df['DILI_Classification'] == 0)],  # New DILI negative molecules
    color='blue',  # Purple color for DILI negative (0)
    alpha=1,  # No transparency for new molecules
    s=30,  # Slightly larger point size for new molecules
    label='DILIst Negative'  # Add label for legend
)

# Set plot title and labels
plt.title('t-SNE Visualization of Molecular ECFP with DILI Classification and Highlighted DILIst Molecules')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')

# Show the plot with legend
plt.legend(title='Classifications')
plt.show()
