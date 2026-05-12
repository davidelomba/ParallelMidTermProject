#include <iostream>
#include <filesystem>
#include <chrono>
#include <omp.h>
#include "preprocessing.h"
#include "morphology.h"
#include "benchmark.h"
namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    std::string raw_data_path = "../data/input";
    std::string grayscale_path = "../data/dataset_grayscale";
    std::string eroded_output_path = "../data/output/eroded";
    std::string dilated_output_path = "../data/output/dilated";
    std::string opening_output_path = "../data/output/opening";
    std::string closing_output_path = "../data/output/closing";

    int max_threads = omp_get_max_threads();
    std::cout << "Numero di thread disponibili: " << max_threads << std::endl;
    omp_set_num_threads(max_threads);




    std::cout << "--- FASE 1: Conversione in scala di grigi ---" << std::endl;
    preprocessDatasetToGrayscale(raw_data_path, grayscale_path);

    // Benchmark per erosione
    run_morphology_benchmark(grayscale_path, eroded_output_path, "Erosione", 3,  erode_sequential, erode_parallel);    

    // Benchmark per dilatazione
    run_morphology_benchmark(grayscale_path, dilated_output_path, "Dilatazione", 3,  dilate_sequential, dilate_parallel);

    // Benchmark per opening
    run_morphology_benchmark(grayscale_path, opening_output_path, "Opening", 3,  opening_sequential, opening_parallel);

    // Benchmark per closing
    run_morphology_benchmark(grayscale_path, closing_output_path, "Closing", 3,  closing_sequential, closing_parallel);
    return 0;
}