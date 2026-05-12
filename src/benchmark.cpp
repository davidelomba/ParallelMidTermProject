#include "benchmark.h"
#include <iostream>
#include <filesystem>
#include <chrono>
#include "morphology.h"

namespace fs = std::filesystem;

// Funzione di utilità per validare che il risultato parallelo sia identico al sequenziale
bool validate(const GrayImage& seq, const GrayImage& par) {
    if (seq.getWidth() != par.getWidth() || seq.getHeight() != par.getHeight()) return false;
    const unsigned char* d_s = seq.getRawData();
    const unsigned char* d_p = par.getRawData();
    for (int i = 0; i < seq.getWidth() * seq.getHeight(); ++i) {
        if (d_s[i] != d_p[i]) return false;
    }
    return true;
}

void run_morphology_benchmark(const std::string& input_path, const std::string& output_path, const std::string& label, int kernel_size, MorphoFunc seq_func, MorphoFunc par_func) 
{
    if (!fs::exists(output_path)) {
        fs::create_directories(output_path);
    }
    std::cout << "\n--- BENCHMARK: " << label << " (Kernel: " << kernel_size << ") ---" << std::endl;

    double total_t_seq = 0, total_t_par = 0;
    int count = 0;

    for (const auto& entry : fs::directory_iterator(input_path)) {
        std::string ext = entry.path().extension().string();
        if (entry.is_regular_file() && (ext == ".png" || ext == ".jpg" || ext == ".jpeg")) {
            GrayImage img;
            if (img.load(entry.path().string())) {
                
                // Misurazione sequenziale
                auto start_seq = std::chrono::high_resolution_clock::now();
                GrayImage res_seq = seq_func(img, kernel_size);
                auto end_seq = std::chrono::high_resolution_clock::now();
                
                // Misurazione parallela
                auto start_par = std::chrono::high_resolution_clock::now();
                GrayImage res_par = par_func(img, kernel_size);
                auto end_par = std::chrono::high_resolution_clock::now();

                // Calcolo speedup e validazione
                std::chrono::duration<double> time_seq = end_seq - start_seq;
                std::chrono::duration<double> time_par = end_par - start_par;

                double speedup = time_seq.count() / time_par.count();
                bool correct = validate(res_seq, res_par);
                
                std::cout << "File: " << entry.path().filename() 
                        << " | T_seq: " << time_seq.count() << "s"
                        << " | T_par: " << time_par.count() << "s"
                        << " | Speedup: " << speedup << "x"
                        << " | Correct: " << (correct ? "SI" : "NO") 
                        << std::endl;

                total_t_seq += time_seq.count();
                total_t_par += time_par.count();
                count++;

                res_par.save(output_path + "/" + label + "_" + entry.path().filename().string());
            }
        }
    }

    if (count > 0) {
        double avg_speedup = total_t_seq / total_t_par;
        std::cout << "\n>>> Conclusione " << label << " <<<" << std::endl;
        std::cout << "Tempo totale Seq: " << total_t_seq << "s" << std::endl;
        std::cout << "Tempo totale Par: " << total_t_par << "s" << std::endl;
        std::cout << "SPEEDUP MEDIO:    " << avg_speedup << "x" << std::endl;
        std::cout << "----------------------------\n" << std::endl;
    } else {
        std::cout << "Nessun file processato per " << label << "." << std::endl;
    }
}