#include <iostream>
#include <string>
#include "preprocessing.h"

int main(int argc, char* argv[]) {
    // 1. Controllo degli argomenti da terminale
    if (argc < 3) {
        std::cerr << "Uso corretto: " << argv[0] << " <cartella_input> <cartella_output>" << std::endl;
        return 1;
    }

    std::string input_folder = argv[1];
    std::string output_folder = argv[2];

    // 2. Esecuzione del pre-processing (carica, converte e salva)
    preprocessDatasetToGrayscale(input_folder, output_folder);

    // 3. Qui aggiungeremo le chiamate per l'Erosione e Dilatazione
    std::cout << "--- Inizio fase di Morfologia Matematica ---" << std::endl;
    std::cout << "Elaborazione in corso..." << std::endl;

    return 0;
}