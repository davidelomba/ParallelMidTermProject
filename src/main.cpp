#include <iostream>
#include <filesystem>
#include <chrono>
#include <omp.h>
#include "morphology.h"
#include "benchmark.h"
namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    // Dimensione dell'immagine da analizzare (corrispondente alla cartella in dataset_grayscale)
    std::string current_scale = "scale_4.0x"; 

    // Definizione dei path di input e output
    std::string grayscale_input_path = "../data/dataset_grayscale/" + current_scale;
    std::string output_base_path = "../data/output/" + current_scale;

    // Configurazione OpenMP
    int max_threads = omp_get_max_threads();
    std::cout << "=== MORPHOLOGY PARALLEL BENCHMARK ===" << std::endl;
    std::cout << "Numero di thread disponibili: " << max_threads << std::endl;
    std::cout << "Analisi su risoluzione: " << current_scale << std::endl;
    omp_set_num_threads(max_threads);

    // Verifica che l'input esista
    if (!fs::exists(grayscale_input_path)) {
        std::cerr << "ERRORE: La cartella " << grayscale_input_path << " non esiste." << std::endl;
        return 1;
    }

    std::cout << "\n--- Avvio Benchmark Morfologici (" << current_scale << ") ---" << std::endl;

    // Benchmark per erosione
    run_morphology_benchmark(grayscale_input_path, output_base_path + "/eroded", "Erosione", 3, erode_sequential, erode_parallel);
    
    // Benchmark per dilatazione
    run_morphology_benchmark(grayscale_input_path, output_base_path + "/dilated", "Dilatazione", 3,  dilate_sequential, dilate_parallel);

    // Benchmark per opening
    run_morphology_benchmark(grayscale_input_path, output_base_path + "/opening", "Opening", 3,  opening_sequential, opening_parallel);

    // Benchmark per closing
    run_morphology_benchmark(grayscale_input_path, output_base_path + "/closing", "Closing", 3,  closing_sequential, closing_parallel);
    return 0;
}