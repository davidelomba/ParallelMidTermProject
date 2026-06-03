import os
import subprocess
import shutil

def setup_bsds(num_images):
    repo_url = "https://github.com/BIDS/BSDS500.git"
    raw_git_dir = "data/raw_bsds_git"

    # Defininzione del percorso di destinazione per le immagini master (da cui poi genereremo le immagini in scala di grigi)    
    master_source_dir = "data/input/scale_1.0x"

    # Controllo se i dati esistono già
    if os.path.exists(master_source_dir) and any(f.endswith('.jpg') for f in os.listdir(master_source_dir)):
        print(f"Dataset originale già presente in {master_source_dir}. Salto il download.")
        return

    print("Scaricando il dataset BSDS500 da GitHub...")

    # Clone shallow per scaricare solo l'ultimo commit
    subprocess.run(["git", "clone", "--depth", "1", repo_url, raw_git_dir])

    src_images_path = os.path.join(raw_git_dir, "BSDS500", "data", "images", "test")
    
    os.makedirs(master_source_dir, exist_ok=True)

    count = 0
    for filename in os.listdir(src_images_path):
        if count >= num_images:
            break
        
        if filename.lower().endswith(".jpg"):
            shutil.copy(os.path.join(src_images_path, filename), master_source_dir)
            count += 1
    
    print(f"Download completato! {count} immagini master copiate in '{master_source_dir}'.")

if __name__ == "__main__":
    NUM_IMAGES = 200
    setup_bsds(num_images=NUM_IMAGES)