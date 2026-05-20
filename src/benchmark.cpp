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
         << res.mean_t1 << "," << res.ci_t1 << "," 
         << res.mean_t2 << "," << res.ci_t2 << "," 
         << res.avg_speedup << "," << res.ci_speedup << "," << efficiency << "\n";
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

    // Fase di warm-up: eseguiamo una singola operazione su una sola immagine per scaldare thread pool e cache
    for (const auto& entry : fs::directory_iterator(input_path)) {
        std::string ext = entry.path().extension().string();
        if (entry.is_regular_file() && (ext == ".png" || ext == ".jpg" || ext == ".jpeg")) {
            GrayImage warm_img;
            if (warm_img.load(entry.path().string())) {
                GrayImage dummy1 = func1(warm_img, kernel_size);
                GrayImage dummy2 = func2(warm_img, kernel_size);
                std::cout << "[WARM-UP] Thread pool OpenMP e cache pronti." << std::endl;
                break;
            }
        }
    }

    const int NUM_RUNS = 5;
    const double T_VALUE = 2.776; // Valore critico T di Student per N=5 (95% di confidenza)
    
    // Vettori che conterranno i tempi totali delle immagini per ciascuna delle 5 run
    std::vector<double> run_totals_1(NUM_RUNS, 0.0);
    std::vector<double> run_totals_2(NUM_RUNS, 0.0);
    int count = 0;

    for (const auto& entry : fs::directory_iterator(input_path)) {
        std::string ext = entry.path().extension().string();
        if (entry.is_regular_file() && (ext == ".png" || ext == ".jpg" || ext == ".jpeg")) {
            GrayImage img;
            if (img.load(entry.path().string())) {
                
                std::vector<double> img_t1(NUM_RUNS);
                std::vector<double> img_t2(NUM_RUNS);
                GrayImage res1, res2;

                // Esecuizione delle 5 run per func1 
                for (int r = 0; r < NUM_RUNS; ++r) {
                    img.load(entry.path().string()); // Cold cache ad ogni run
                    auto start1 = std::chrono::high_resolution_clock::now();
                    res1 = func1(img, kernel_size);
                    auto end1 = std::chrono::high_resolution_clock::now();
                    img_t1[r] = std::chrono::duration<double>(end1 - start1).count();
                }

                // Esecuzione delle 5 run per func2
                for (int r = 0; r < NUM_RUNS; ++r) {
                    img.load(entry.path().string()); // Cold cache ad ogni run
                    auto start2 = std::chrono::high_resolution_clock::now();
                    res2 = func2(img, kernel_size);
                    auto end2 = std::chrono::high_resolution_clock::now();
                    img_t2[r] = std::chrono::duration<double>(end2 - start2).count();
                }

                // Validazione del risultato di func2 contro func1 (devono essere identici)
                bool correct = validate(res1, res2);
                
                // Accumulo dei tempi totali per ciascuna run e somma dei tempi per il calcolo della media a livello di immagine
                double sum_img_t1 = 0;
                double sum_img_t2 = 0;
                for (int r = 0; r < NUM_RUNS; ++r) {
                    run_totals_1[r] += img_t1[r];
                    run_totals_2[r] += img_t2[r];
                    sum_img_t1 += img_t1[r];
                    sum_img_t2 += img_t2[r];
                }

                // Stampa del tempo medio e dello speed up della singola immagine
                double img_mean1 = sum_img_t1 / NUM_RUNS;
                double img_mean2 = sum_img_t2 / NUM_RUNS;
                std::cout << "File: " << entry.path().filename() 
                          << " | T1 (Media): " << img_mean1 << "s"
                          << " | T2 (Media): " << img_mean2 << "s"
                          << " | Speedup: " << (img_mean1 / img_mean2) << "x"
                          << " | Correct: " << (correct ? "SI" : "NO") 
                          << std::endl;

                count++;
                res2.save(output_path + "/" + label + "_" + entry.path().filename().string());
            }
        }
    }

    if (count > 0) {
        
        // Calcolo delle medie dei tempi totali
        double mean_t1 = 0.0, mean_t2 = 0.0;
        for (int r = 0; r < NUM_RUNS; ++r) {
            mean_t1 += run_totals_1[r];
            mean_t2 += run_totals_2[r];
        }
        mean_t1 /= NUM_RUNS;
        mean_t2 /= NUM_RUNS;

        // Calcolo della deviazione standard per i tempi totali
        double var_t1 = 0.0, var_t2 = 0.0;
        for (int r = 0; r < NUM_RUNS; ++r) {
            var_t1 += (run_totals_1[r] - mean_t1) * (run_totals_1[r] - mean_t1);
            var_t2 += (run_totals_2[r] - mean_t2) * (run_totals_2[r] - mean_t2);
        }
        double std_dev1 = std::sqrt(var_t1 / (NUM_RUNS - 1));
        double std_dev2 = std::sqrt(var_t2 / (NUM_RUNS - 1));

        // Calcolo degli intervalli di confidenza per i tempi totali
        double ci_t1 = T_VALUE * (std_dev1 / std::sqrt(NUM_RUNS));
        double ci_t2 = T_VALUE * (std_dev2 / std::sqrt(NUM_RUNS));

        // Calcolo dello speedup medio e del suo intervallo di confidenza
        std::vector<double> run_speedups(NUM_RUNS);
        double avg_speedup = 0.0;
        for (int r = 0; r < NUM_RUNS; ++r) {
            run_speedups[r] = run_totals_1[r] / run_totals_2[r];
            avg_speedup += run_speedups[r];
        }
        avg_speedup /= NUM_RUNS;

        double var_speedup = 0.0;
        for (int r = 0; r < NUM_RUNS; ++r) {
            var_speedup += (run_speedups[r] - avg_speedup) * (run_speedups[r] - avg_speedup);
        }
        double std_dev_speedup = std::sqrt(var_speedup / (NUM_RUNS - 1));
        double ci_speedup = T_VALUE * (std_dev_speedup / std::sqrt(NUM_RUNS));

        std::cout << "\n>>> Conclusione " << label << " <<<" << std::endl;
        std::cout << "Tempo totale medio 1: " << mean_t1 << "s ± " << ci_t1 << "s" << std::endl;
        std::cout << "Tempo totale medio 2: " << mean_t2 << "s ± " << ci_t2 << "s" << std::endl;
        std::cout << "SPEEDUP MEDIO:        " << avg_speedup << "x ± " << ci_speedup << "x" << std::endl;
        std::cout << "----------------------------\n" << std::endl;

        return {mean_t1, ci_t1, mean_t2, ci_t2, avg_speedup, ci_speedup};
    }
    
    std::cout << "Nessun file processato per " << label << "." << std::endl;
    return {0, 0, 0, 0, 0, 0};
}

void run_strong_scaling_test(const std::string& scale, const std::vector<int>& thread_configs) {
    std::string base_dir = "../results/strong_scaling/" + scale;
    if (!fs::exists(base_dir)) {
        fs::create_directories(base_dir);
    }

    std::string csv_path = base_dir + "/strong_scaling_results.csv";
    std::ofstream csv_file(csv_path);
    
    // Header aggiornato con colonne dedicate per gli intervalli di confidenza (CI)
    csv_file << "Test,Scale,Threads,Operation,TimeSeq_Mean,TimeSeq_CI,TimePar_Mean,TimePar_CI,Speedup_Mean,Speedup_CI,Efficiency\n";

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
    
    // Header aggiornato
    csv_file << "Test,Scale,Threads,Operation,TimeSeq_Mean,TimeSeq_CI,TimePar_Mean,TimePar_CI,Speedup_Mean,Speedup_CI,Efficiency\n";

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
    
    // Header aggiornato
    csv_file << "Test,Scale,KernelSize,Threads,Operation,TimeStandard_Mean,TimeStandard_CI,TimeSeparable_Mean,TimeSeparable_CI,SeparableSpeedup_Mean,SeparableSpeedup_CI\n";

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
                     << r_ero.mean_t1 << "," << r_ero.ci_t1 << "," << r_ero.mean_t2 << "," << r_ero.ci_t2 << "," << r_ero.avg_speedup << "," << r_ero.ci_speedup << "\n";
            
            // --- TEST 2: DILATAZIONE ---
            std::this_thread::sleep_for(std::chrono::seconds(2));
            auto r_dil = run_morphology_benchmark(input_path, output_base + "/dilated", "Dilatazione", k, dilate_parallel, dilate_separable_parallel);
            csv_file << "Separable Evaluation," << scale << "," << k << "," << threads << ",Dilatazione," 
                     << r_dil.mean_t1 << "," << r_dil.ci_t1 << "," << r_dil.mean_t2 << "," << r_dil.ci_t2 << "," << r_dil.avg_speedup << "," << r_dil.ci_speedup << "\n";

            // --- TEST 3: OPENING ---
            std::this_thread::sleep_for(std::chrono::seconds(2));
            auto r_ope = run_morphology_benchmark(input_path, output_base + "/opening", "Opening", k, opening_parallel, opening_separable_parallel);
            csv_file << "Separable Evaluation," << scale << "," << k << "," << threads << ",Opening," 
                     << r_ope.mean_t1 << "," << r_ope.ci_t1 << "," << r_ope.mean_t2 << "," << r_ope.ci_t2 << "," << r_ope.avg_speedup << "," << r_ope.ci_speedup << "\n";

            // --- TEST 4: CLOSING ---
            std::this_thread::sleep_for(std::chrono::seconds(2));
            auto r_clo = run_morphology_benchmark(input_path, output_base + "/closing", "Closing", k, closing_parallel, closing_separable_parallel);
            csv_file << "Separable Evaluation," << scale << "," << k << "," << threads << ",Closing," 
                     << r_clo.mean_t1 << "," << r_clo.ci_t1 << "," << r_clo.mean_t2 << "," << r_clo.ci_t2 << "," << r_clo.avg_speedup << "," << r_clo.ci_speedup << "\n";
            
            csv_file.flush();
        }
    }

    csv_file.close();
    std::cout << "\n[OK] Test Separable Parallelism completato con successo." << std::endl;
}

void run_pipeline_test(const std::vector<std::string>& scales, const std::vector<int>& kernel_sizes) {    
    std::string base_dir = "../results/pipeline";
    if (!fs::exists(base_dir)) {
        fs::create_directories(base_dir);
    }

    std::string csv_path = base_dir + "/pipeline_results.csv";
    std::ofstream csv_file(csv_path);
    
    // Header aggiornato
    csv_file << "Test,Scale,KernelSize,Threads,Operation,TimeStandard_Mean,TimeStandard_CI,TimePipeline_Mean,TimePipeline_CI,PipelineSpeedup_Mean,PipelineSpeedup_CI\n";

    std::cout << "=== PIPELINE PARALLELISM TEST ===" << std::endl;
    std::cout << "CSV: " << csv_path << std::endl;

    omp_set_num_threads(2);

    for (const auto& scale : scales) {
        std::string input_path = "../data/dataset_grayscale/" + scale;
        std::string output_base = "../data/output/" + scale;

        if (!fs::exists(input_path)) {
            std::cerr << "[AVVISO] Cartella non trovata " << input_path << std::endl;
            continue;
        }

        for (int k : kernel_sizes) {
            std::cout << "\n>>> Scale: " << scale << " | Kernel Size: " << k << "x" << k << " <<<" << std::endl;

            // --- TEST 1: OPENING ---
            std::this_thread::sleep_for(std::chrono::seconds(2));
            auto r_ope = run_morphology_benchmark(input_path, output_base + "/opening", "Opening", k, opening_parallel, opening_pipeline);
            csv_file << "Pipeline Evaluation," << scale << "," << k << ",2,Opening," 
                     << r_ope.mean_t1 << "," << r_ope.ci_t1 << "," << r_ope.mean_t2 << "," << r_ope.ci_t2 << "," << r_ope.avg_speedup << "," << r_ope.ci_speedup << "\n";

            // --- TEST 2: CLOSING ---
            std::this_thread::sleep_for(std::chrono::seconds(2));
            auto r_clo = run_morphology_benchmark(input_path, output_base + "/closing", "Closing", k, closing_parallel, closing_pipeline);
            csv_file << "Pipeline Evaluation," << scale << "," << k << ",2,Closing," 
                     << r_clo.mean_t1 << "," << r_clo.ci_t1 << "," << r_clo.mean_t2 << "," << r_clo.ci_t2 << "," << r_clo.avg_speedup << "," << r_clo.ci_speedup << "\n";
            
            csv_file.flush();
        }
    }

    csv_file.close();
    std::cout << "\n[OK] Test Pipeline Parallelism completato con successo." << std::endl;
}