import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurazione dei percorsi
CSV_FOLDER = "results/optimal_evaluation" 
FILE_NAME = "optimal_vs_sequential_results.csv"
FULL_PATH = os.path.join(CSV_FOLDER, FILE_NAME)
OUTPUT_FOLDER = "graphs"

# Creazione automatica della cartella di output se non esiste
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Caricamento dati
if not os.path.exists(FULL_PATH):
    raise FileNotFoundError(f"Non ho trovato il file in: {os.path.abspath(FULL_PATH)}")

df = pd.read_csv(FULL_PATH)

sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 6))

# Disegna le linee di speedup
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

plt.title("Speedup dell'Algoritmo Ottimale vs Sequenziale\n(Erosione, 4 Threads)", fontsize=14, pad=15)
plt.xlabel("Dimensione del Kernel", fontsize=12)
plt.ylabel("Speedup", fontsize=12)

unique_kernels = sorted(df['KernelSize'].unique())
plt.xticks(unique_kernels)

handles, labels = plt.gca().get_legend_handles_labels()
clean_labels = [l.replace('scale_', '').replace('x', '0x') for l in labels]
plt.legend(handles, clean_labels, title="Risoluzione", fontsize=10, title_fontsize=11)

plt.tight_layout()

# Salva il grafico 
save_name = 'optimal_speedup.png'
save_path = os.path.join(OUTPUT_FOLDER, save_name)
plt.savefig(save_path, dpi=300)
print(f"Grafico salvato con successo in: {save_path}")

plt.show()
plt.close()