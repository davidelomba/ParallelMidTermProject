import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_FOLDER = "results/optimal_evaluation" 
FILE_NAME = "optimal_vs_sequential_results.csv"
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
    x='KernelSize', 
    y='OptimalSpeedup_Mean', 
    hue='Scale', 
    marker='o', 
    linewidth=2.5,
    markersize=8,
    palette="magma" 
)

plt.title("Optimal vs. Sequential Algorithm Speedup\n(Erosion, 4 Threads)", fontsize=14, pad=15)
plt.xlabel("Kernel Size", fontsize=12)
plt.ylabel("Speedup", fontsize=12)

unique_kernels = sorted(df['KernelSize'].unique())
plt.xticks(unique_kernels)

handles, labels = plt.gca().get_legend_handles_labels()
clean_labels = [l.replace('scale_', '').replace('x', '0x') for l in labels]
plt.legend(handles, clean_labels, title="Resolution", fontsize=10, title_fontsize=11)

plt.tight_layout()

save_name = 'optimal_speedup.png'
save_path = os.path.join(OUTPUT_FOLDER, save_name)
plt.savefig(save_path, dpi=300)
print(f"Graph successfully saved in: {save_path}")
plt.close()