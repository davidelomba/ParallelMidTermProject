#include "preprocessing.h"
#include "GrayImage.h"
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

void preprocessDatasetToGrayscale(const std::string& input_folder, const std::string& output_folder) {
    if (!fs::exists(output_folder)) {
        fs::create_directories(output_folder);
    }

    std::cout << "Inizio elaborazione batch con classe GrayImage..." << std::endl;

    for (const auto& entry : fs::directory_iterator(input_folder)) {
        if (entry.is_regular_file()) {
            std::string filepath = entry.path().string();
            std::string out_filepath = output_folder + "/" + entry.path().stem().string() + ".png";

            GrayImage img;
            if (img.load(filepath)) {
                if (img.save(out_filepath)) {
                    std::cout << "  -> Elaborata: " << entry.path().filename() << std::endl;
                }
            } else {
                std::cerr << "Errore nel caricamento di: " << filepath << std::endl;
            }
        }
    }
}