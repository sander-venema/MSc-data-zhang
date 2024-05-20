import os
import pandas as pd
import matplotlib.pyplot as plt

csv_dir = 'csv/mean/'

plt.figure(figsize=(10, 6))

for file_name in os.listdir(csv_dir):
    if file_name.endswith('genloss.csv'):
        file_path = os.path.join(csv_dir, file_name)
        df = pd.read_csv(file_path)
        plt.plot(df['Step'], df['Value'], marker='', linestyle='-')
    
plt.xlabel("Epoch")
plt.ylabel("Mean Generator Loss")
plt.title("Mean Generator Loss across epoch")
plt.legend(["WGAN Augmented", "WGAN GP Augmented", "WGAN GP"])
plt.savefig("plots/mean_generator_loss.png")
plt.clf()

for file_name in os.listdir(csv_dir):
    if file_name.endswith('disloss.csv'):
        file_path = os.path.join(csv_dir, file_name)
        df = pd.read_csv(file_path)
        plt.plot(df['Step'], df['Value'], marker='', linestyle='-')

plt.xlabel("Epoch")
plt.ylabel("Mean Critic Loss")
plt.title("Mean Critic Loss across epoch")
plt.legend(["WGAN Augmented", "WGAN GP Augmented", "WGAN GP"])
plt.savefig("plots/mean_critic_loss.png")
plt.clf()