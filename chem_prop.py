import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem, DataStructs
from rdkit.Chem import rdchem, Descriptors, rdMolDescriptors, AllChem, rdFingerprintGenerator
from rdkit.Chem import rdFingerprintGenerator as fpg

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Add a column to differentiate between the two datasets
df1['Source'] = 'DILIst'
df2['Source'] = 'Inferred'

# Concatenate the two dataframes for comparison
combined_df = pd.concat([df1, df2])

# List of chemical properties to plot
chemical_properties = ['HBA', 'HBD', 'TPSA', 'LogP', 'num_rot_bonds', 'Molecular_Weight']

# Set the style of the seaborn plots
sns.set(style="whitegrid")

# Create a grid of violin plots for each chemical property
for property in chemical_properties:
    plt.figure(figsize=(10, 6), dpi=300)
    sns.violinplot(data=combined_df, x='DILI_Classification', y=property, hue='Source', split=True, palette="Set2", inner='quartile')
    plt.title(f'Comparison of {property} by DILI Classification and Molecule Source')
    plt.xlabel('DILI Classification')
    plt.ylabel(property)
    plt.legend(title='Source')
    plt.show()
