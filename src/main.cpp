#include <iostream>
#include <filesystem>
#include "GrayImage.h"
#include "preprocessing.h"
#include "morphology.h"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    std::string raw_data_path = "../data/input";
    std::string grayscale_path = "../data/dataset_grayscale";
    std::string eroded_output_path = "../data/output/eroded";
    std::string dilated_output_path = "../data/output/dilated";


    std::cout << "--- FASE 1: Conversione in scala di grigi ---" << std::endl;
    preprocessDatasetToGrayscale(raw_data_path, grayscale_path);

    std::cout << "--- FASE 2: Applicazione Erosione ---" << std::endl;
    
    if (!fs::exists(eroded_output_path)) {
        fs::create_directories(eroded_output_path);
    }

    for (const auto& entry : fs::directory_iterator(grayscale_path)) {
        if (entry.is_regular_file()) {
            std::string filepath = entry.path().string();
            std::string filename = entry.path().filename().string();

            GrayImage img;
            if (img.load(filepath)) {
                GrayImage eroded = erode_sequential(img, 3);

                std::string out_path = eroded_output_path + "/eroded_" + filename;
                if (eroded.save(out_path)) {
                    std::cout << "Successo: " << filename << " erosa e salvata." << std::endl;
                }
            }
        }
    }

    std::cout << "\nProcesso completato!" << std::endl;


    std::cout << "--- FASE 3: Applicazione Dilatazione ---" << std::endl;
    
    if (!fs::exists(dilated_output_path)) {
        fs::create_directories(dilated_output_path);
    }

    for (const auto& entry : fs::directory_iterator(grayscale_path)) {
        if (entry.is_regular_file()) {
            std::string filepath = entry.path().string();
            std::string filename = entry.path().filename().string();

            GrayImage img;
            if (img.load(filepath)) {
                GrayImage dilated = dilate_sequential(img, 3);

                std::string out_path = dilated_output_path + "/dilated_" + filename;
                if (dilated.save(out_path)) {
                    std::cout << "Successo: " << filename << " dilatata e salvata." << std::endl;
                }
            }
        }
    }

    std::cout << "\nProcesso completato!" << std::endl;
    return 0;
}