#include "benchmark.h"
#include <iostream>
#include <filesystem>
#include <chrono>
#include <fstream>
#include <vector>
#include <thread>
#include <omp.h>
#include "morphology.h"

namespace fs = std::filesystem;

void save_benchmark_to_csv(std::ofstream& file, std::string law_name, std::string scale, int threads, std::string op, BenchResult res) {
    double efficiency = (threads > 0) ? (res.avg_speedup / threads) : 0.0;
    file << law_name << "," << scale << "," << threads << "," << op << "," 
         << res.total_t_seq << "," << res.total_t_par << "," << res.avg_speedup << "," << efficiency << "\n";
    file.flush();
}

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

BenchResult run_morphology_benchmark(const std::string& input_path, const std::string& output_path, const std::string& label, int kernel_size, MorphoFunc seq_func, MorphoFunc par_func) {
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
        return {total_t_seq, total_t_par, avg_speedup};
    }
    std::cout << "Nessun file processato per " << label << "." << std::endl;
    return {0, 0, 0};
    
}

void run_strong_scaling_test(const std::string& scale, const std::vector<int>& thread_configs) {

    std::string base_dir = "../results/strong_scaling/" + scale;
    if (!fs::exists(base_dir)) {
        fs::create_directories(base_dir);
    }

    std::string csv_path = base_dir + "/strong_scaling_results.csv";

    std::ofstream csv_file(csv_path);
    csv_file << "Law,Scale,Threads,Operation,TimeSeq,TimePar,Speedup,Efficiency\n";

    std::string input_path = "../data/dataset_grayscale/" + scale;
    std::string output_base = "../data/output/" + scale;

    std::cout << "=== STRONG SCALING TEST: " << scale << " ===" << std::endl;
    std::cout << "CSV: " << csv_path << std::endl;

    for (int t : thread_configs) {
        std::cout << "\n>>> Configurazione: " << t << " Threads <<<" << std::endl;

        // Fai riposare la CPU per 2 secondi per abbassare la temperatura
        std::this_thread::sleep_for(std::chrono::seconds(3));

        omp_set_num_threads(t);

        auto r_ero = run_morphology_benchmark(input_path, output_base + "/eroded", "Erosione", 3, erode_sequential, erode_parallel);
        save_benchmark_to_csv(csv_file, "Strong Scaling", scale, t, "Erosione", r_ero);

        // Fai riposare la CPU per 2 secondi per abbassare la temperatura
        std::this_thread::sleep_for(std::chrono::seconds(2));

        auto r_dil = run_morphology_benchmark(input_path, output_base + "/dilated", "Dilatazione", 3, dilate_sequential, dilate_parallel);
        save_benchmark_to_csv(csv_file, "Strong Scaling", scale, t, "Dilatazione", r_dil);

        // Fai riposare la CPU per 2 secondi per abbassare la temperatura
        std::this_thread::sleep_for(std::chrono::seconds(2));

        auto r_ope = run_morphology_benchmark(input_path, output_base + "/opening", "Opening", 3, opening_sequential, opening_parallel);
        save_benchmark_to_csv(csv_file, "Strong Scaling", scale, t, "Opening", r_ope);

        // Fai riposare la CPU per 2 secondi per abbassare la temperatura
        std::this_thread::sleep_for(std::chrono::seconds(2));

        auto r_clo = run_morphology_benchmark(input_path, output_base + "/closing", "Closing", 3, closing_sequential, closing_parallel);
        save_benchmark_to_csv(csv_file, "Strong Scaling", scale, t, "Closing", r_clo);

    }

    csv_file.close();
    std::cout << "\n[OK] Test Strong Scaling completato." << std::endl;
}

void run_weak_scaling_test() {
    std::string base_dir = "../results/weak_scaling";
    if (!fs::exists(base_dir)) fs::create_directories(base_dir);

    std::string csv_path = base_dir + "/weak_scaling_results.csv";
    std::ofstream csv_file(csv_path);
    csv_file << "Law,Scale,Threads,Operation,TimeSeq,TimePar,Speedup,Efficiency\n";


    std::vector<std::pair<int, std::string>> config = {
        {1, "scale_1.0x"},
        {4, "scale_2.0x"},
        {16, "scale_4.0x"}
    };

    std::cout << "=== WEAK SCALING TEST ===" << std::endl;

    for (auto const& [t, scale] : config) {
        std::cout << "\n>>> WEAK SCALING: " << t << " Threads su " << scale << " <<<" << std::endl;
        
        std::string input_path = "../data/dataset_grayscale/" + scale;
        std::string output_base = "../data/output/" + scale;

        if (!fs::exists(input_path)) {
            std::cout << "Skipping " << scale << " (cartella non trovata)" << std::endl;
            continue;
        }

        // Fai riposare la CPU per 2 secondi per abbassare la temperatura
        std::this_thread::sleep_for(std::chrono::seconds(3));

        omp_set_num_threads(t);

        // Eseguiamo le operazioni (usiamo la stessa funzione di salvataggio dell'altra volta)
        auto r_ero = run_morphology_benchmark(input_path, output_base + "/eroded", "Erosione", 3, erode_sequential, erode_parallel);
        save_benchmark_to_csv(csv_file, "Weak Scaling", scale, t, "Erosione", r_ero);

        // Fai riposare la CPU per 2 secondi per abbassare la temperatura
        std::this_thread::sleep_for(std::chrono::seconds(2));

        auto r_dil = run_morphology_benchmark(input_path, output_base + "/dilated", "Dilatazione", 3, dilate_sequential, dilate_parallel);
        save_benchmark_to_csv(csv_file, "Weak Scaling", scale, t, "Dilatazione", r_dil);

        // Fai riposare la CPU per 2 secondi per abbassare la temperatura
        std::this_thread::sleep_for(std::chrono::seconds(2));

        auto r_ope = run_morphology_benchmark(input_path, output_base + "/opening", "Opening", 3, opening_sequential, opening_parallel);
        save_benchmark_to_csv(csv_file, "Weak Scaling", scale, t, "Opening", r_ope);

        // Fai riposare la CPU per 2 secondi per abbassare la temperatura
        std::this_thread::sleep_for(std::chrono::seconds(2));

        auto r_clo = run_morphology_benchmark(input_path, output_base + "/closing", "Closing", 3, closing_sequential, closing_parallel);
        save_benchmark_to_csv(csv_file, "Weak Scaling", scale, t, "Closing", r_clo);


    }

    csv_file.close();
    std::cout << "\n[OK] Test Weak Scaling completato." << std::endl;
}