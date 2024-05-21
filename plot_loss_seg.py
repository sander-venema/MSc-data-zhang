import os
import pandas as pd
import matplotlib.pyplot as plt

csv_dir = 'csv/segmentation/'

plt.figure(figsize=(10, 6))

for file_name in os.listdir(csv_dir):
    if file_name.endswith('Metrics_IoU.csv'):
        file_path = os.path.join(csv_dir, file_name)
        df = pd.read_csv(file_path)
        plt.plot(df['Step'], df['Value'], marker='', linestyle='-')
    
plt.xlabel("Epoch")
plt.ylabel("IoU Score")
plt.title("Validation IoU Score across epoch")
plt.legend(["ResNet101 1e-4",
            "ResNet101 5e-4",
            "UNet ResNext101",
            "UNet VGG16bn",
            "UNet VGG16",
            "UNet VGG19bn",
            "UNet VGG19"
            ])
plt.savefig("plots/segmentation_iou.png")
plt.clf()
