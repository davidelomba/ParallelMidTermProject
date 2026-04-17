#include <iostream>
#include <filesystem>
#include "GrayImage.h"
#include "preprocessing.h"
#include "morphology.h"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    std::string raw_data_path = "../data/input";
    std::string grayscale_path = "../data/dataset_grayscale";
    std::string final_output_path = "../data/output";

    std::cout << "--- FASE 1: Conversione in scala di grigi ---" << std::endl;
    preprocessDatasetToGrayscale(raw_data_path, grayscale_path);

std::cout << "--- FASE 2: Applicazione Erosione ---" << std::endl;
    
    if (!fs::exists(final_output_path)) {
        fs::create_directories(final_output_path);
    }

    for (const auto& entry : fs::directory_iterator(grayscale_path)) {
        if (entry.is_regular_file()) {
            std::string filepath = entry.path().string();
            std::string filename = entry.path().filename().string();

            GrayImage img;
            if (img.load(filepath)) {
                GrayImage eroded = erode_sequential(img, 3);

                std::string out_path = final_output_path + "/eroded_" + filename;
                if (eroded.save(out_path)) {
                    std::cout << "Successo: " << filename << " erosa e salvata." << std::endl;
                }
            }
        }
    }

    std::cout << "\nProcesso completato!" << std::endl;
    return 0;
}