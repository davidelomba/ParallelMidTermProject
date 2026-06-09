import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurazione dei percorsi
CSV_FOLDER = "results/weak_scaling" 
FILE_NAME = "weak_scaling_results.csv"
FULL_PATH = os.path.join(CSV_FOLDER, FILE_NAME)
OUTPUT_FOLDER = "graphs"

# Creazione automatica della cartella di output se non esiste
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Caricamento dati
if not os.path.exists(FULL_PATH):
    raise FileNotFoundError(f"Non ho trovato il file in: {os.path.abspath(FULL_PATH)}")

df = pd.read_csv(FULL_PATH)

# Imposta lo stile grafico base per tutti i grafici
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

# Linea di Speedup Ideale
plt.plot([1, 16], [1, 16], color='black', linestyle='--', linewidth=1.5, label='Speedup Ideale')

plt.title("Weak Scaling\nAndamento dello Speedup", fontsize=14, pad=15)
plt.xlabel("Numero di Thread", fontsize=12)
plt.ylabel("Speedup", fontsize=12)
plt.xticks([1, 4, 16])
plt.xlim(0.5, 16.5) 
plt.ylim(0, 16.5)

plt.legend(title="Operazione", fontsize=10, title_fontsize=11)
plt.tight_layout()

save_path_speedup = os.path.join(OUTPUT_FOLDER, 'weak_scaling_speedup.png')
plt.savefig(save_path_speedup, dpi=300)
print(f"Grafico salvato con successo in: {save_path_speedup}")

plt.close() 


plt.figure(figsize=(10, 6))

sns.lineplot(
    data=df, 
    x='Threads', 
    y='TimePar_Mean', 
    hue='Operation', 
    marker='o', 
    linewidth=2.5,
    markersize=8
)

plt.title("Weak Scaling\nAndamento del Tempo di Esecuzione dell'Algoritmo Parallelo", fontsize=14, pad=15)
plt.xlabel("Numero di Thread", fontsize=12)
plt.ylabel("Tempo di Esecuzione (secondi)", fontsize=12)
plt.xticks([1, 4, 16])
plt.xlim(0.5, 16.5) 
plt.ylim(0, df['TimePar_Mean'].max() + 1.0)

plt.legend(title="Operazione", fontsize=10, title_fontsize=11)
plt.tight_layout()

save_path_time = os.path.join(OUTPUT_FOLDER, 'weak_scaling_time.png')
plt.savefig(save_path_time, dpi=300)
print(f"Grafico salvato con successo in: {save_path_time}")

plt.close()