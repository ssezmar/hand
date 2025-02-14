import pandas as pd

# Load the dataset
df = pd.read_csv('../data/raw/dataset.csv')

# Check the completeness of the dataset
total_samples = len(df)
bent_arms = df[df['label'] == 1].shape[0]
extended_arms = df[df['label'] == 0].shape[0]

# Print the results
print(f'Total samples: {total_samples}')
print(f'Bent arms: {bent_arms}')
print(f'Extended arms: {extended_arms}')