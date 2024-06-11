import os
import pandas as pd
import matplotlib.pyplot as plt

csv_dir = 'csv/mean/appendix'

plt.figure(figsize=(10, 6))

for file_name in os.listdir(csv_dir):
    if file_name.endswith('genloss.csv'):
        file_path = os.path.join(csv_dir, file_name)
        df = pd.read_csv(file_path)
        plt.plot(df['Step'], df['Value'], marker='', linestyle='-')
    
plt.xlabel("Epoch")
plt.ylabel("Mean Generator Loss")
plt.title("Mean Generator Loss across epoch")
plt.legend(["WassAug16", "WassAug64", "WassGPaug16", "WassGPaug64"])
plt.savefig("plots/mean_generator_loss_appendix.png")
plt.clf()

for file_name in os.listdir(csv_dir):
    if file_name.endswith('disloss.csv'):
        file_path = os.path.join(csv_dir, file_name)
        df = pd.read_csv(file_path)
        plt.plot(df['Step'], df['Value'], marker='', linestyle='-')

plt.xlabel("Epoch")
plt.ylabel("Mean Critic Loss")
plt.title("Mean Critic Loss across epoch")
plt.legend(["WassAug16", "WassAug64", "WassGPaug16", "WassGPaug64"])
plt.savefig("plots/mean_critic_loss_appendix.png")
plt.clf()