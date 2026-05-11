#include <iostream>
#include <filesystem>
#include <chrono>
#include <omp.h>
#include "GrayImage.h"
#include "preprocessing.h"
#include "morphology.h"

namespace fs = std::filesystem;

// Funzione di utilità per validare che il risultato parallelo sia identico al sequenziale
bool validate(const GrayImage& seq, const GrayImage& par) {
    if (seq.getWidth() != par.getWidth() || seq.getHeight() != par.getHeight()) return false;
    const unsigned char* data_seq = seq.getRawData();
    const unsigned char* data_par = par.getRawData();
    for (int i = 0; i < seq.getWidth() * seq.getHeight(); ++i) {
        if (data_seq[i] != data_par[i]) return false;
    }
    return true;
}

int main(int argc, char* argv[]) {
    std::string raw_data_path = "../data/input";
    std::string grayscale_path = "../data/dataset_grayscale";
    std::string eroded_output_path = "../data/output/eroded";
    std::string dilated_output_path = "../data/output/dilated";
    std::string opening_output_path = "../data/output/opening";
    std::string closing_output_path = "../data/output/closing";

    int max_threads = omp_get_max_threads();
    std::cout << "Il programma utilizzerà un massimo di " << max_threads << " thread." << std::endl;


    std::cout << "--- FASE 1: Conversione in scala di grigi ---" << std::endl;
    //preprocessDatasetToGrayscale(raw_data_path, grayscale_path);

    std::cout << "--- FASE 2: Applicazione Erosione ---" << std::endl;
    
    if (!fs::exists(eroded_output_path)) {
        fs::create_directories(eroded_output_path);
    }

    std::cout << "--- BENCHMARK: Erosione (Sequenziale vs Parallelo) ---" << std::endl;

    double total_t_seq = 0;
    double total_t_par = 0;
    int count = 0;

    for (const auto& entry : fs::directory_iterator(grayscale_path)) {
        std::string ext = entry.path().extension().string();
        if (entry.is_regular_file() && (ext == ".png" || ext == ".jpg" || ext == ".jpeg")) {
            GrayImage img;
            if (img.load(entry.path().string())) {
                
                // 1. Misurazione Sequenziale
                auto start_seq = std::chrono::high_resolution_clock::now();
                GrayImage res_seq = erode_sequential(img, 3);
                auto end_seq = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> time_seq = end_seq - start_seq;

                // 2. Misurazione Parallela
                auto start_par = std::chrono::high_resolution_clock::now();
                GrayImage res_par = erode_parallel(img, 3);
                auto end_par = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> time_par = end_par - start_par;

                // 3. Calcolo Speedup e Validazione
                double speedup = time_seq.count() / time_par.count();
                bool is_correct = validate(res_seq, res_par);

                std::cout << "File: " << entry.path().filename() 
                          << " | T_seq: " << time_seq.count() << "s"
                          << " | T_par: " << time_par.count() << "s"
                          << " | Speedup: " << speedup << "x"
                          << " | Correct: " << (is_correct ? "SI" : "NO") 
                          << std::endl;

                total_t_seq += time_seq.count();
                total_t_par += time_par.count();
                count++;

                res_par.save(eroded_output_path + "/eroded_" + entry.path().filename().string());          
            }
        }
    }

    if (count > 0) {
    double avg_speedup = total_t_seq / total_t_par;
    std::cout << "\n>>> CONCLUSIONE EROSIONE <<<" << std::endl;
    std::cout << "Tempo totale Seq: " << total_t_seq << "s" << std::endl;
    std::cout << "Tempo totale Par: " << total_t_par << "s" << std::endl;
    std::cout << "SPEEDUP MEDIO:    " << avg_speedup << "x" << std::endl;
    std::cout << "----------------------------\n" << std::endl;
    } else {
        std::cout << "Nessun file processato per l'erosione." << std::endl;
    }

    std::cout << "\nProcesso completato!" << std::endl;


    std::cout << "--- FASE 3: Applicazione Dilatazione ---" << std::endl;
    
    if (!fs::exists(dilated_output_path)) {
        fs::create_directories(dilated_output_path);
    }

    std::cout << "\n--- BENCHMARK: Dilatazione (Sequenziale vs Parallelo) ---" << std::endl;

    total_t_seq = 0;
    total_t_par = 0;
    count = 0;


    for (const auto& entry : fs::directory_iterator(grayscale_path)) {
        std::string ext = entry.path().extension().string();
        if (entry.is_regular_file() && (ext == ".png" || ext == ".jpg" || ext == ".jpeg")) {
            GrayImage img;

            if (img.load(entry.path().string())) {

                
                // 1. Misurazione Sequenziale
                auto start_seq = std::chrono::high_resolution_clock::now();
                GrayImage res_seq = dilate_sequential(img, 3);
                auto end_seq = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> time_seq = end_seq - start_seq;

                // 2. Misurazione Parallela
                auto start_par = std::chrono::high_resolution_clock::now();
                GrayImage res_par = dilate_parallel(img, 3);
                auto end_par = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> time_par = end_par - start_par;

                // 3. Calcolo Speedup e Validazione
                double speedup = time_seq.count() / time_par.count();
                bool is_correct = validate(res_seq, res_par);

                std::cout << "File: " << entry.path().filename() 
                        << " | T_seq: " << time_seq.count() << "s"
                        << " | T_par: " << time_par.count() << "s"
                        << " | Speedup: " << speedup << "x"
                        << " | Correct: " << (is_correct ? "SI" : "NO") 
                        << std::endl;

                total_t_seq += time_seq.count();
                total_t_par += time_par.count();
                count++;
                
                res_par.save(dilated_output_path + "/dilated_" + entry.path().filename().string());
            }
        }
    }

    if (count > 0) {
        double avg_speedup = total_t_seq / total_t_par;
        std::cout << "\n>>> CONCLUSIONE DILATAZIONE <<<" << std::endl;
        std::cout << "Tempo totale Seq: " << total_t_seq << "s" << std::endl;
        std::cout << "Tempo totale Par: " << total_t_par << "s" << std::endl;
        std::cout << "SPEEDUP MEDIO:    " << avg_speedup << "x" << std::endl;
        std::cout << "----------------------------\n" << std::endl;
    } else {
        std::cout << "Nessun file processato per la dilatazione." << std::endl;
    }

    std::cout << "\nProcesso completato!" << std::endl;

    std::cout << "--- FASE 4: Applicazione Opening ---" << std::endl;
    if (!fs::exists(opening_output_path)) {
        fs::create_directories(opening_output_path);
    }
    std::cout << "\n--- BENCHMARK: Opening (Sequenziale vs Parallelo) ---" << std::endl;

    total_t_seq = 0;
    total_t_par = 0;
    count = 0;

    for (const auto& entry : fs::directory_iterator(grayscale_path)) {
        std::string ext = entry.path().extension().string();
        if (entry.is_regular_file() && (ext == ".png" || ext == ".jpg" || ext == ".jpeg")) {
            GrayImage img;
            if (img.load(entry.path().string())) {
                
                // 1. Misurazione Sequenziale
                auto start_seq = std::chrono::high_resolution_clock::now();
                GrayImage res_seq = opening_sequential(img, 3);
                auto end_seq = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> time_seq = end_seq - start_seq;

                // 2. Misurazione Parallela
                auto start_par = std::chrono::high_resolution_clock::now();
                GrayImage res_par = opening_parallel(img, 3);
                auto end_par = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> time_par = end_par - start_par;

                // 3. Calcolo Speedup e Validazione
                double speedup = time_seq.count() / time_par.count();
                bool is_correct = validate(res_seq, res_par);

                total_t_seq += time_seq.count();
                total_t_par += time_par.count();
                count++;

                std::cout << "File: " << entry.path().filename() 
                        << " | T_seq: " << time_seq.count() << "s"
                        << " | T_par: " << time_par.count() << "s"
                        << " | Speedup: " << speedup << "x"
                        << " | Correct: " << (is_correct ? "SI" : "NO") 
                        << std::endl;
                res_par.save(opening_output_path + "/opening_" + entry.path().filename().string());
            }
        }
    }
    if (count > 0) {
        double avg_speedup = total_t_seq / total_t_par;
        std::cout << "\n>>> CONCLUSIONE OPENING <<<" << std::endl;
        std::cout << "Tempo totale Seq: " << total_t_seq << "s" << std::endl;
        std::cout << "Tempo totale Par: " << total_t_par << "s" << std::endl;
        std::cout << "SPEEDUP MEDIO:    " << avg_speedup << "x" << std::endl;
        std::cout << "----------------------------\n" << std::endl;
    } else {
        std::cout << "Nessun file processato per l'opening." << std::endl;
    }

    std::cout << "\nProcesso completato!" << std::endl;

    std::cout << "--- FASE 5: Applicazione Closing ---" << std::endl;
    if (!fs::exists(closing_output_path)) {
        fs::create_directories(closing_output_path);
    }

    std::cout << "\n--- BENCHMARK: Closing (Sequenziale vs Parallelo) ---" << std::endl;

    total_t_seq = 0;
    total_t_par = 0;
    count = 0;

    for (const auto& entry : fs::directory_iterator(grayscale_path)) {
        std::string ext = entry.path().extension().string();
        if (entry.is_regular_file() && (ext == ".png" || ext == ".jpg" || ext == ".jpeg")) {
            GrayImage img;
            if (img.load(entry.path().string())) {
                
                // 1. Misurazione Sequenziale
                auto start_seq = std::chrono::high_resolution_clock::now();
                GrayImage res_seq = closing_sequential(img, 3);
                auto end_seq = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> time_seq = end_seq - start_seq;

                // 2. Misurazione Parallela
                auto start_par = std::chrono::high_resolution_clock::now();
                GrayImage res_par = closing_parallel(img, 3);
                auto end_par = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> time_par = end_par - start_par;

                // 3. Calcolo Speedup e Validazione
                double speedup = time_seq.count() / time_par.count();
                bool is_correct = validate(res_seq, res_par);

                total_t_seq += time_seq.count();
                total_t_par += time_par.count();
                count++;

                std::cout << "File: " << entry.path().filename() 
                        << " | T_seq: " << time_seq.count() << "s"
                        << " | T_par: " << time_par.count() << "s"
                        << " | Speedup: " << speedup << "x"
                        << " | Correct: " << (is_correct ? "SI" : "NO") 
                        << std::endl;
                res_par.save(closing_output_path + "/closing_" + entry.path().filename().string());
            }
        }
    } 
    if (count > 0) {
        double avg_speedup = total_t_seq / total_t_par;
        std::cout << "\n>>> CONCLUSIONE CLOSING <<<" << std::endl;
        std::cout << "Tempo totale Seq: " << total_t_seq << "s" << std::endl;
        std::cout << "Tempo totale Par: " << total_t_par << "s" << std::endl;
        std::cout << "SPEEDUP MEDIO:    " << avg_speedup << "x" << std::endl;
        std::cout << "----------------------------\n" << std::endl;
    } else {
        std::cout << "Nessun file processato per il closing." << std::endl;
    }
    return 0;
}