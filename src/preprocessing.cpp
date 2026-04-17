#include "preprocessing.h"
#include <iostream>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace fs = std::filesystem;

void preprocessDatasetToGrayscale(const std::string& input_folder, const std::string& output_folder) {
    if (!fs::exists(output_folder)) {
        fs::create_directories(output_folder);
        std::cout << "Cartella di output creata: " << output_folder << std::endl;
    }

    std::cout << "Inizio elaborazione batch: conversione in scala di grigi..." << std::endl;

    for (const auto& entry : fs::directory_iterator(input_folder)) {
        if (entry.is_regular_file()) {
            std::string filepath = entry.path().string();
            std::string stem = entry.path().stem().string();
            std::string out_filepath = output_folder + "/" + stem + ".png";

            int width, height, channels;
            
            unsigned char *img = stbi_load(filepath.c_str(), &width, &height, &channels, 1);
            
            if (img == NULL) {
                std::cerr << "Errore: Impossibile caricare " << filepath << std::endl;
                continue; 
            }

            int stride_in_bytes = width * 1;
            int success = stbi_write_png(out_filepath.c_str(), width, height, 1, img, stride_in_bytes);
            
            if (success) {
                std::cout << "  -> Convertita e salvata: " << stem << ".png" << std::endl;
            } else {
                std::cerr << "  -> Errore nel salvataggio di: " << out_filepath << std::endl;
            }

            stbi_image_free(img);
        }
    }
    std::cout << "Pre-processing completato con successo!\n" << std::endl;
}