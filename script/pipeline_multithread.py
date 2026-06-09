import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurazione dei percorsi
CSV_FOLDER = "results/pipeline_multithread" 
FILE_NAME = "pipeline_multithread_results.csv"
FULL_PATH = os.path.join(CSV_FOLDER, FILE_NAME)
OUTPUT_FOLDER = "graphs"

# Creazione automatica della cartella di output se non esiste
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Caricamento dati
if not os.path.exists(FULL_PATH):
    raise FileNotFoundError(f"Non ho trovato il file in: {os.path.abspath(FULL_PATH)}")

df = pd.read_csv(FULL_PATH)

# Filtra i dati per includere sia Opening che Closing
df_filtered = df[df['Operation'].isin(['Opening', 'Closing'])]

# Imposta lo stile grafico base
sns.set_theme(style="whitegrid")

# Identifica le scale uniche presenti nei dati
unique_scales = df_filtered['Scale'].unique()

for current_scale in unique_scales:
    # Filtra i dati per la scala corrente
    df_scale = df_filtered[df_filtered['Scale'] == current_scale]
    
    plt.figure(figsize=(10, 6))
    
    # Disegna le linee di speedup per le operazioni
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
    
    # Configurazione del titolo dinamica in base alla scala corrente
    scale_label = current_scale.replace('scale_', '').replace('x', '0x')
    plt.title(f"Speedup della Pipeline Produttore-Consumatore\n(Opening vs Closing) - Scala {scale_label}", fontsize=14, pad=15)
    plt.xlabel("Numero di Thread", fontsize=12)
    plt.ylabel("Speedup", fontsize=12)
    plt.xticks([2, 4, 8])
    
    # Imposta i limiti dell'asse Y in modo dinamico per la scala corrente
    plt.ylim(df_scale['Speedup_Mean'].min() - 0.1, df_scale['Speedup_Mean'].max() + 0.2)
    
    plt.legend(title="Operazione", fontsize=10, title_fontsize=11)
    plt.tight_layout()
    
    # Salva il grafico 
    save_name = f'pipeline_speedup_{current_scale}.png'
    save_path = os.path.join(OUTPUT_FOLDER, save_name)
    plt.savefig(save_path, dpi=300)
    print(f"Grafico salvato con successo in: {save_path}")
    
    plt.show()
    plt.close()