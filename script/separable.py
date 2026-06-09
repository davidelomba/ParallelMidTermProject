import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurazione dei percorsi
CSV_FOLDER = "results/separable" 
FILE_NAME = "separable_results.csv"
FULL_PATH = os.path.join(CSV_FOLDER, FILE_NAME)
OUTPUT_FOLDER = "graphs"

# Creazione automatica della cartella di output se non esiste
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Caricamento dati
if not os.path.exists(FULL_PATH):
    raise FileNotFoundError(f"Non ho trovato il file in: {os.path.abspath(FULL_PATH)}")

df = pd.read_csv(FULL_PATH)

# Imposta lo stile grafico base
sns.set_theme(style="whitegrid")

scales_to_plot = df['Scale'].unique()

for current_scale in scales_to_plot:
    # Filtra i dati per la scala corrente e limitiamo a 4 Thread
    df_filtered = df[(df['Scale'] == current_scale) & (df['Threads'] == 4)]
    
    # Salta l'iterazione se non ci sono dati per questa combinazione
    if df_filtered.empty:
        continue
    
    plt.figure(figsize=(12, 7)) 

    # Crea il grafico a barre raggruppate
    barplot = sns.barplot(
        data=df_filtered, 
        x='KernelSize', 
        y='SeparableSpeedup_Mean', 
        hue='Operation',
        palette="viridis"
    )

    # Aggiunge i valori numerici sopra le barre
    for p in barplot.patches:
        height = p.get_height()
        # Evita di annotare eventuali barre vuote o a zero
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

    # Configurazione scritte e titoli
    scale_label = str(current_scale).replace('scale_', '').replace('x', '0x') 
    plt.title(f"Impatto dell'Algoritmo Separabile 1D vs 2D Standard\n(Scale {scale_label}, 4 Thread)", fontsize=14, pad=15)
    plt.xlabel("Dimensione del Kernel", fontsize=12)
    plt.ylabel("Speedup", fontsize=12)
    
    plt.ylim(0, df_filtered['SeparableSpeedup_Mean'].max() + 1.5)

    plt.legend(title="Operazione", fontsize=10, title_fontsize=11)
    plt.tight_layout()

    # Salva il grafico con il nome della scala rilevata
    save_name = f'separable_speedup_{current_scale}.png'
    save_path = os.path.join(OUTPUT_FOLDER, save_name)
    plt.savefig(save_path, dpi=300)
    print(f"Grafico salvato con successo in: {save_path}")
    
    plt.close()