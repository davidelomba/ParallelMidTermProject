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

void save_benchmark_to_csv(std::ofstream& file, std::string Test_name, std::string scale, int threads, std::string op, BenchResult res) {
    double efficiency = (threads > 0) ? (res.avg_speedup / threads) : 0.0;
    file << Test_name << "," << scale << "," << threads << "," << op << "," 
         << res.total_t_func1 << "," << res.total_t_func2 << "," << res.avg_speedup << "," << efficiency << "\n";
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

BenchResult run_morphology_benchmark(const std::string& input_path, const std::string& output_path, const std::string& label, int kernel_size, MorphoFunc func1, MorphoFunc func2) {
    if (!fs::exists(output_path)) {
        fs::create_directories(output_path);
    }
    std::cout << "\n--- BENCHMARK: " << label << " (Kernel: " << kernel_size << ") ---" << std::endl;

    double total_t_1 = 0;
    double total_t_2 = 0;
    int count = 0;

    for (const auto& entry : fs::directory_iterator(input_path)) {
        std::string ext = entry.path().extension().string();
        if (entry.is_regular_file() && (ext == ".png" || ext == ".jpg" || ext == ".jpeg")) {
            GrayImage img;
            if (img.load(entry.path().string())) {
                
                auto start1 = std::chrono::high_resolution_clock::now();
                GrayImage res1 = func1(img, kernel_size);
                auto end1 = std::chrono::high_resolution_clock::now();

                // Ricarica l'immagine per fare svuotare la cache e garantire condizioni simili per il secondo test
                img.load(entry.path().string());
                
                auto start2 = std::chrono::high_resolution_clock::now();
                GrayImage res2 = func2(img, kernel_size);
                auto end2 = std::chrono::high_resolution_clock::now();

                // Calcolo speedup e validazione
                std::chrono::duration<double> time1 = end1 - start1;
                std::chrono::duration<double> time2 = end2 - start2;

                double speedup = time1.count() / time2.count();
                bool correct = validate(res1, res2);
                
                std::cout << "File: " << entry.path().filename() 
                        << " | T1: " << time1.count() << "s"
                        << " | T2: " << time2.count() << "s"
                        << " | Speedup: " << speedup << "x"
                        << " | Correct: " << (correct ? "SI" : "NO") 
                        << std::endl;

                total_t_1 += time1.count();
                total_t_2 += time2.count();
                count++;

                res2.save(output_path + "/" + label + "_" + entry.path().filename().string());
            }
        }
    }

    if (count > 0) {
        double avg_speedup = total_t_1 / total_t_2;
        std::cout << "\n>>> Conclusione " << label << " <<<" << std::endl;
        std::cout << "Tempo totale 1: " << total_t_1 << "s" << std::endl;
        std::cout << "Tempo totale 2: " << total_t_2 << "s" << std::endl;
        std::cout << "SPEEDUP MEDIO:    " << avg_speedup << "x" << std::endl;
        std::cout << "----------------------------\n" << std::endl;
        return {total_t_1, total_t_2, avg_speedup};
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
    csv_file << "Test,Scale,Threads,Operation,TimeSeq,TimePar,Speedup,Efficiency\n";

    std::string input_path = "../data/dataset_grayscale/" + scale;
    std::string output_base = "../data/output/" + scale;

    std::cout << "=== STRONG SCALING TEST: " << scale << " ===" << std::endl;
    std::cout << "CSV: " << csv_path << std::endl;

    for (int t : thread_configs) {
        std::cout << "\n>>> Configurazione: " << t << " Threads <<<" << std::endl;

        std::this_thread::sleep_for(std::chrono::seconds(3));

        omp_set_num_threads(t);

        auto r_ero = run_morphology_benchmark(input_path, output_base + "/eroded", "Erosione", 3, erode_sequential, erode_parallel);
        save_benchmark_to_csv(csv_file, "Strong Scaling", scale, t, "Erosione", r_ero);

        std::this_thread::sleep_for(std::chrono::seconds(2));

        auto r_dil = run_morphology_benchmark(input_path, output_base + "/dilated", "Dilatazione", 3, dilate_sequential, dilate_parallel);
        save_benchmark_to_csv(csv_file, "Strong Scaling", scale, t, "Dilatazione", r_dil);

        std::this_thread::sleep_for(std::chrono::seconds(2));
        auto r_ope = run_morphology_benchmark(input_path, output_base + "/opening", "Opening", 3, opening_sequential, opening_parallel);
        save_benchmark_to_csv(csv_file, "Strong Scaling", scale, t, "Opening", r_ope);

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
    csv_file << "Test,Scale,Threads,Operation,TimeSeq,TimePar,Speedup,Efficiency\n";


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

        std::this_thread::sleep_for(std::chrono::seconds(3));

        omp_set_num_threads(t);

        auto r_ero = run_morphology_benchmark(input_path, output_base + "/eroded", "Erosione", 3, erode_sequential, erode_parallel);
        save_benchmark_to_csv(csv_file, "Weak Scaling", scale, t, "Erosione", r_ero);

        std::this_thread::sleep_for(std::chrono::seconds(2));

        auto r_dil = run_morphology_benchmark(input_path, output_base + "/dilated", "Dilatazione", 3, dilate_sequential, dilate_parallel);
        save_benchmark_to_csv(csv_file, "Weak Scaling", scale, t, "Dilatazione", r_dil);

        std::this_thread::sleep_for(std::chrono::seconds(2));

        auto r_ope = run_morphology_benchmark(input_path, output_base + "/opening", "Opening", 3, opening_sequential, opening_parallel);
        save_benchmark_to_csv(csv_file, "Weak Scaling", scale, t, "Opening", r_ope);

        std::this_thread::sleep_for(std::chrono::seconds(2));

        auto r_clo = run_morphology_benchmark(input_path, output_base + "/closing", "Closing", 3, closing_sequential, closing_parallel);
        save_benchmark_to_csv(csv_file, "Weak Scaling", scale, t, "Closing", r_clo);


    }

    csv_file.close();
    std::cout << "\n[OK] Test Weak Scaling completato." << std::endl;
}


void run_separable_test(const std::vector<std::string>& scales, int threads, const std::vector<int>& kernel_sizes){    
    std::string base_dir = "../results/separable";
    if (!fs::exists(base_dir)) {
        fs::create_directories(base_dir);
    }

    std::string csv_path = base_dir + "/separable_results.csv";
    std::ofstream csv_file(csv_path);
    
    csv_file << "Test,Scale,KernelSize,Threads,Operation,TotalTimeStandard,TotalTimeSeparable,SeparableSpeedup\n";

    std::cout << "=== SEPARABLE PARALLELISM TEST ===" << std::endl;
    std::cout << "Thread: " << threads << " | CSV: " << csv_path << std::endl;

    omp_set_num_threads(threads);

    for (const auto& scale : scales) {
        std::string input_path = "../data/dataset_grayscale/" + scale;
        std::string output_base = "../data/output/" + scale;

        if (!fs::exists(input_path)) {
            std::cerr << "[AVVISO] Cartella non trovata " << input_path << std::endl;
            continue;
        }

        for (int k : kernel_sizes) {
            std::cout << "\n>>> Scale: " << scale << " | Kernel Size: " << k << "x" << k << " <<<" << std::endl;

            // --- TEST 1: EROSIONE ---
            std::this_thread::sleep_for(std::chrono::seconds(3));
            auto r_ero = run_morphology_benchmark(input_path, output_base + "/eroded", "Erosione", k, erode_parallel, erode_separable_parallel);
            csv_file << "Separable Evaluation," << scale << "," << k << "," << threads << ",Erosione," 
                     << r_ero.total_t_func1 << "," << r_ero.total_t_func2 << "," << r_ero.avg_speedup << "\n";
            
            // --- TEST 2: DILATAZIONE ---
            std::this_thread::sleep_for(std::chrono::seconds(2));
            auto r_dil = run_morphology_benchmark(input_path, output_base + "/dilated", "Dilatazione", k, dilate_parallel, dilate_separable_parallel);
            csv_file << "Separable Evaluation," << scale << "," << k << "," << threads << ",Dilatazione," 
                     << r_dil.total_t_func1 << "," << r_dil.total_t_func2 << "," << r_dil.avg_speedup << "\n";

            // --- TEST 3: OPENING ---
            std::this_thread::sleep_for(std::chrono::seconds(2));
            auto r_ope = run_morphology_benchmark(input_path, output_base + "/opening", "Opening", k, opening_parallel, opening_separable_parallel);
            csv_file << "Separable Evaluation," << scale << "," << k << "," << threads << ",Opening," 
                     << r_ope.total_t_func1 << "," << r_ope.total_t_func2 << "," << r_ope.avg_speedup << "\n";

            // --- TEST 4: CLOSING ---
            std::this_thread::sleep_for(std::chrono::seconds(2));
            auto r_clo = run_morphology_benchmark(input_path, output_base + "/closing", "Closing", k, closing_parallel, closing_separable_parallel);
            csv_file << "Separable Evaluation," << scale << "," << k << "," << threads << ",Closing," 
                     << r_clo.total_t_func1 << "," << r_clo.total_t_func2 << "," << r_clo.avg_speedup << "\n";
            
            csv_file.flush();
        }
    }

    csv_file.close();
    std::cout << "\n[OK] Test Separable Parallelism completato con successo." << std::endl;
}
