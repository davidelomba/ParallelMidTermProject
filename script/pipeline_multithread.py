import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_FOLDER = "results/pipeline_multithread" 
FILE_NAME = "pipeline_multithread_results.csv"
FULL_PATH = os.path.join(CSV_FOLDER, FILE_NAME)
OUTPUT_FOLDER = "graphs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

if not os.path.exists(FULL_PATH):
    raise FileNotFoundError(f"File not found in: {os.path.abspath(FULL_PATH)}")

df = pd.read_csv(FULL_PATH)

df_filtered = df[df['Operation'].isin(['Opening', 'Closing'])]

sns.set_theme(style="whitegrid")

unique_scales = df_filtered['Scale'].unique()

for current_scale in unique_scales:
    df_scale = df_filtered[df_filtered['Scale'] == current_scale]
    
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(
        data=df_scale, 
        x='Threads', 
        y='Speedup_Mean', 
        hue='Operation', 
        marker='o', 
        linewidth=2.5,
        markersize=8,
        palette="muted"
    )
    
    scale_label = current_scale.replace('scale_', '').replace('x', '0x')
    plt.title(f"Producer-Consumer Pipeline Speedup\n(Opening vs. Closing) - Scale {scale_label}", fontsize=14, pad=15)
    plt.xlabel("Number of Threads", fontsize=12)
    plt.ylabel("Speedup", fontsize=12)
    plt.xticks([2, 4, 8])
    
    plt.ylim(df_scale['Speedup_Mean'].min() - 0.1, df_scale['Speedup_Mean'].max() + 0.2)
    
    plt.legend(title="Operation", fontsize=10, title_fontsize=11)
    plt.tight_layout()
    
    save_name = f'pipeline_speedup_{current_scale}.png'
    save_path = os.path.join(OUTPUT_FOLDER, save_name)
    plt.savefig(save_path, dpi=300)
    print(f"Graph successfully saved in: {save_path}")
    
    plt.close()