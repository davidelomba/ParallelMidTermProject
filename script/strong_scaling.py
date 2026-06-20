import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_FOLDER = "results/strong_scaling/scale_4.0x" 
FILE_NAME = "strong_scaling_results.csv"
FULL_PATH = os.path.join(CSV_FOLDER, FILE_NAME) 
OUTPUT_FOLDER = "graphs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

if not os.path.exists(FULL_PATH):
    raise FileNotFoundError(f"File not found in: {os.path.abspath(FULL_PATH)}")

df = pd.read_csv(FULL_PATH)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

sns.lineplot(
    data=df, 
    x='Threads', 
    y='Speedup_Mean', 
    hue='Operation', 
    marker='o', 
    linewidth=2.5,
    markersize=8
)

plt.plot([1, 8], [1, 8], color='black', linestyle='--', linewidth=1.5, label='Ideal Speedup')

plt.title("Strong Scaling (Scale 4.0x)\n", fontsize=14, pad=15)
plt.xlabel("Number of Threads", fontsize=12)
plt.ylabel("Speedup", fontsize=12)
plt.xticks([1, 2, 4, 8, 16, 32])
plt.xlim(1, 32)

plt.legend(title="Operation / Reference", fontsize=10, title_fontsize=11)
plt.tight_layout()

save_path = os.path.join(OUTPUT_FOLDER, 'strong_scaling.png')
plt.savefig(save_path, dpi=300)
print(f"Graph successfully saved in: {save_path}")
