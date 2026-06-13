import os
from PIL import Image

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def process_to_grayscale():
    source_dir = os.path.join(BASE_DIR, "data", "input", "scale_1.0x")
    target_dir = os.path.join(BASE_DIR, "data", "dataset_grayscale", "scale_1.0x")

    if not os.path.exists(source_dir):
        print(f"Errore: sorgente {source_dir} non trovata")
        return

    os.makedirs(target_dir, exist_ok=True)
    print(f"Inizio conversione in scala di grigi...")

    images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for filename in images:
        with Image.open(os.path.join(source_dir, filename)) as img:
            gray_img = img.convert('L')
            target_path = os.path.join(target_dir, os.path.splitext(filename)[0] + ".png")
            gray_img.save(target_path)
            
    print(f"Completato! {len(images)} immagini salvate in {target_dir}")

if __name__ == "__main__":
    process_to_grayscale()