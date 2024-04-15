import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import numpy as np
import re

# Path to the directory containing CSV files
csv_dir = 'csv/mean/'

pattern = r".*realmean\.csv$"

plt.figure(figsize=(10, 6))

# Loop through files in the directory
for file_name in os.listdir(csv_dir):
    if re.match(pattern, file_name):
        print(file_name)
        file_path = os.path.join(csv_dir, file_name)
        df = pd.read_csv(file_path)

        # Smooth the data using a cubic spline
        xnew = np.linspace(df['Step'].min(), df['Step'].max(), 300)
        spl = make_interp_spline(df['Step'], df['Value'], k=3)
        smoothed_values = spl(xnew)
        plt.plot(xnew, smoothed_values, marker='', linestyle='-')

plt.xlabel("Step")
plt.ylabel("Mean Real Validation Accuracy")
plt.title("Smoothed mean real validation accuracy across steps")
plt.legend(["WassAug32", "WassGP", "WassGPaug32"])
plt.grid(True)
plt.savefig("plots/realmean_values_smoothed.png")
plt.show()
