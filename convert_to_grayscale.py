import os
from PIL import Image

def process_to_grayscale():
    source_dir = "data/input/scale_1.0x"
    target_dir = "data/dataset_grayscale/scale_1.0x"

    if not os.path.exists(source_dir):
        print(f"Errore: sorgente {source_dir} non trovata")
        return

    os.makedirs(target_dir, exist_ok=True)
    print(f"Inizio conversione in scala di grigi...")

    images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for filename in images:
        # Apertura dell'immagine a colori
        with Image.open(os.path.join(source_dir, filename)) as img:
            # Conversione in scala di grigi (L = Luminance)
            gray_img = img.convert('L')
            
            # Salvataggio dell'immagine in scala di grigi con estensione .png
            target_path = os.path.join(target_dir, os.path.splitext(filename)[0] + ".png")
            gray_img.save(target_path)
            
    print(f"Completato! {len(images)} immagini salvate in {target_dir}")

if __name__ == "__main__":
    process_to_grayscale()