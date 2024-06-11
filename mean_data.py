import os
import pandas as pd

csv_dir = 'csv/wassGPaug16_5e-05_loss/'

dfs = []

name = ''

for file_name in os.listdir(csv_dir):
    if file_name.endswith('Generator.csv'):
        file_path = os.path.join(csv_dir, file_name)
        df = pd.read_csv(file_path)
        df = df.iloc[:, 1:]  # Drop the first column (timestamp)
        dfs.append(df)
        name = file_name

combined_df = pd.concat(dfs, axis=0)

mean_accuracy = combined_df.groupby('Step')['Value'].mean().reset_index()

mean_accuracy.to_csv('csv/mean/wassGPaug16_5e-05_loss_genloss.csv', index=False)
