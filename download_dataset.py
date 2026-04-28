import os
import subprocess
import shutil

def setup_bsds():
    repo_url = "https://github.com/BIDS/BSDS500.git"
    target_dir = "data/raw_bsds"
    final_input_dir = "data/input"

    if os.path.exists(final_input_dir):
        files = [f for f in os.listdir(final_input_dir) if not f.startswith('.')]
        if len(files) > 0:
            print("Dataset già presente in data/input")
            return

    if not os.path.exists("data"):
        os.makedirs("data")

    print("Scaricando il dataset BSDS500...")
    subprocess.run(["git", "clone", repo_url, target_dir])

    src_images = os.path.join(target_dir, "BSDS500", "data", "images", "test")
    
    if not os.path.exists(final_input_dir):
        os.makedirs(final_input_dir)

    for filename in os.listdir(src_images):
        if filename.endswith(".jpg"):
            shutil.copy(os.path.join(src_images, filename), final_input_dir)
    
    print(f"Setup completato! Le immagini sono state copiate in '{final_input_dir}'.")

if __name__ == "__main__":
    setup_bsds()