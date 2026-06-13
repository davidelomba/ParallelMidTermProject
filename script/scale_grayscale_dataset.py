import os
from PIL import Image

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def scale_grayscale(factors):
    base_gray_path = os.path.join(BASE_DIR, "data", "dataset_grayscale")
    source_dir = os.path.join(base_gray_path, "scale_1.0x")

    if not os.path.exists(source_dir):
        print(f"Errore: sorgente {source_dir} non trovata. Esegui prima convert_to_grayscale.py")
        return

    images = [f for f in os.listdir(source_dir) if f.endswith('.png')]

    for f in factors:
        if f == 1.0: continue
        
        target_label = f"scale_{f}x"
        target_dir = os.path.join(base_gray_path, target_label)
        os.makedirs(target_dir, exist_ok=True)
        
        print(f"Scaling {target_label} in corso...")

        for img_name in images:
            dest_path = os.path.join(target_dir, img_name)
            if not os.path.exists(dest_path):
                with Image.open(os.path.join(source_dir, img_name)) as img:
                    new_size = (int(img.width * f), int(img.height * f))
                    resized = img.resize(new_size, Image.Resampling.LANCZOS)
                    resized.save(dest_path)
                    
    print("\nScaling completato per tutti i fattori.")

if __name__ == "__main__":
    test_factors = [1.0, 2.0, 4.0]
    scale_grayscale(test_factors)