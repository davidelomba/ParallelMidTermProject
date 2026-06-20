import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_FOLDER = "results/separable" 
FILE_NAME = "separable_results.csv"
FULL_PATH = os.path.join(CSV_FOLDER, FILE_NAME)
OUTPUT_FOLDER = "graphs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

if not os.path.exists(FULL_PATH):
    raise FileNotFoundError(f"File not found in: {os.path.abspath(FULL_PATH)}")

df = pd.read_csv(FULL_PATH)

sns.set_theme(style="whitegrid")

scales_to_plot = df['Scale'].unique()

for current_scale in scales_to_plot:
    df_filtered = df[(df['Scale'] == current_scale) & (df['Threads'] == 4)]
    
    if df_filtered.empty:
        continue
    
    plt.figure(figsize=(12, 7)) 

    barplot = sns.barplot(
        data=df_filtered, 
        x='KernelSize', 
        y='SeparableSpeedup_Mean', 
        hue='Operation',
        palette="viridis"
    )

    for p in barplot.patches:
        height = p.get_height()
        if pd.notnull(height) and height > 0:
            barplot.annotate(
                format(height, '.1f') + "x", 
                (p.get_x() + p.get_width() / 2., height), 
                ha='center', va='center', 
                xytext=(0, 8), 
                textcoords='offset points',
                fontsize=9,
                weight='bold'
            )

    scale_label = str(current_scale).replace('scale_', '').replace('x', '0x') 
    plt.title(f"1D Separable vs. Standard 2D Algorithm Impact\n(Scale {scale_label}, 4 Threads)", fontsize=14, pad=15)
    plt.xlabel("Kernel Size", fontsize=12)
    plt.ylabel("Speedup", fontsize=12)
    
    plt.ylim(0, df_filtered['SeparableSpeedup_Mean'].max() + 1.5)

    plt.legend(title="Operation", fontsize=10, title_fontsize=11)
    plt.tight_layout()

    save_name = f'separable_speedup_{current_scale}.png'
    save_path = os.path.join(OUTPUT_FOLDER, save_name)
    plt.savefig(save_path, dpi=300)
    print(f"Graph successfully saved in: {save_path}")
    
    plt.close()