import os
import pandas as pd

# Path to the directory containing CSV files
csv_dir = 'csv/wassaug32_5e-05_loss/'

# List to store dataframes from each CSV file
dfs = []

name = ''
# Loop through files in the directory
for file_name in os.listdir(csv_dir):
    if file_name.endswith('Generator.csv'):
        file_path = os.path.join(csv_dir, file_name)
        df = pd.read_csv(file_path)
        # Assuming the CSV files have headers, adjust skiprows if necessary
        df = df.iloc[:, 1:]  # Drop the first column (timestamp)
        dfs.append(df)
        name = file_name

# Concatenate all dataframes along axis=0 (stack vertically)
combined_df = pd.concat(dfs, axis=0)

# Calculate the mean accuracy for each step
mean_accuracy = combined_df.groupby('Step')['Value'].mean().reset_index()

# Save the averaged data to a new CSV file
mean_accuracy.to_csv('csv/mean/wassaug32_5e-05_loss_genloss.csv', index=False)
