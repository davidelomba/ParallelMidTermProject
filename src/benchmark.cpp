#include "benchmark.h"
#include <iostream>
#include <filesystem>
#include <chrono>
#include <fstream>
#include <vector>
#include <thread>
#include <omp.h>
#include <cmath>
#include "morphology.h"


namespace fs = std::filesystem;

// Funzione che salva i risultati del benchmark (tempi, intervalli di confidenza, speedup ed efficienza) in formato CSV
void save_benchmark_to_csv(std::ofstream& file, std::string Test_name, std::string scale, int threads, std::string op, BenchResult res) {
    double efficiency = (threads > 0) ? (res.avg_speedup / threads) : 0.0;
    file << Test_name << "," << scale << "," << threads << "," << op << "," 
         << res.mean_t1 << "," << res.ci_t1 << "," 
         << res.mean_t2 << "," << res.ci_t2 << "," 
         << res.avg_speedup << "," << res.ci_speedup << "," << efficiency << "\n";
    file.flush();
}

// Funzione di utilità per verificare che il risultato delle funzioni in esame sia lo stesso
bool validate(const GrayImage& seq, const GrayImage& par) {
    if (seq.getWidth() != par.getWidth() || seq.getHeight() != par.getHeight()) return false;
    const unsigned char* d_s = seq.getRawData();
    const unsigned char* d_p = par.getRawData();
    for (int i = 0; i < seq.getWidth() * seq.getHeight(); ++i) {
        if (d_s[i] != d_p[i]) return false;
    }
    return true;
}

// Esegue il benchmark per una specifica operazione morfologica su un intero dataset, calcolando medie, speedup e intervalli di confidenza
BenchResult run_morphology_benchmark(const std::string& input_path, const std::string& output_path, const std::string& label, int kernel_size, MorphoFunc func1, MorphoFunc func2) {
    if (!fs::exists(output_path)) {
        fs::create_directories(output_path);
    }
    std::cout << "\n--- BENCHMARK: " << label << " (Kernel: " << kernel_size << ") ---" << std::endl;

    // Fase di warm-up: esegue una singola operazione su una sola immagine per scaldare thread pool e cache
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

// Funzione per il test di strong scaling, valutando l'impatto dell'aumento dei thread su una dimensione dell'immagine fissa
void run_strong_scaling_test(const std::string& scale, const std::vector<int>& thread_configs) {
    std::string base_dir = "../results/strong_scaling/" + scale;
    if (!fs::exists(base_dir)) {
        fs::create_directories(base_dir);
    }

    std::string csv_path = base_dir + "/strong_scaling_results.csv";
    std::ofstream csv_file(csv_path);
    
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

// Funzione per il test di weak scaling, nella quale si aumenta in modo proporzionale sia la dimensione del problema che il numero di thread
void run_weak_scaling_test() {
    std::string base_dir = "../results/weak_scaling";
    if (!fs::exists(base_dir)) fs::create_directories(base_dir);

    std::string csv_path = base_dir + "/weak_scaling_results.csv";
    std::ofstream csv_file(csv_path);
    
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

// Funzione per valutare le prestazioni dei kernel morfologici separabili 1D rispetto ai corrispondenti approcci standard 2D in parallelo
void run_separable_test(const std::vector<std::string>& scales, const std::vector<int>& thread_configs, const std::vector<int>& kernel_sizes){    
    std::string base_dir = "../results/separable";
    if (!fs::exists(base_dir)) {
        fs::create_directories(base_dir);
    }

    std::string csv_path = base_dir + "/separable_results.csv";
    std::ofstream csv_file(csv_path);
    
    // Header
    csv_file << "Test,Scale,KernelSize,Threads,Operation,TimeStandard_Mean,TimeStandard_CI,TimeSeparable_Mean,TimeSeparable_CI,SeparableSpeedup_Mean,SeparableSpeedup_CI\n";

    std::cout << "=== SEPARABLE PARALLELISM TEST ===" << std::endl;
    std::cout << "CSV: " << csv_path << std::endl;

    for (const auto& scale : scales) {
        std::string input_path = "../data/dataset_grayscale/" + scale;
        std::string output_base = "../data/output/" + scale;

        if (!fs::exists(input_path)) {
            std::cerr << "[AVVISO] Cartella non trovata " << input_path << std::endl;
            continue;
        }

        for (int t : thread_configs) {
            std::cout << "\n>>> Configurazione: " << t << " Threads <<<" << std::endl;
            omp_set_num_threads(t);

            for (int k : kernel_sizes) {
                std::cout << "\n>>> Scale: " << scale << " | Threads: " << t << " | Kernel Size: " << k << "x" << k << " <<<" << std::endl;

                // --- TEST 1: EROSIONE ---
                std::this_thread::sleep_for(std::chrono::seconds(3));
                auto r_ero = run_morphology_benchmark(input_path, output_base + "/eroded", "Erosione", k, erode_parallel, erode_separable_parallel);
                csv_file << "Separable Evaluation," << scale << "," << k << "," << t << ",Erosione," 
                         << r_ero.mean_t1 << "," << r_ero.ci_t1 << "," << r_ero.mean_t2 << "," << r_ero.ci_t2 << "," << r_ero.avg_speedup << "," << r_ero.ci_speedup << "\n";
                
                // --- TEST 2: DILATAZIONE ---
                std::this_thread::sleep_for(std::chrono::seconds(2));
                auto r_dil = run_morphology_benchmark(input_path, output_base + "/dilated", "Dilatazione", k, dilate_parallel, dilate_separable_parallel);
                csv_file << "Separable Evaluation," << scale << "," << k << "," << t << ",Dilatazione," 
                         << r_dil.mean_t1 << "," << r_dil.ci_t1 << "," << r_dil.mean_t2 << "," << r_dil.ci_t2 << "," << r_dil.avg_speedup << "," << r_dil.ci_speedup << "\n";

                // --- TEST 3: OPENING ---
                std::this_thread::sleep_for(std::chrono::seconds(2));
                auto r_ope = run_morphology_benchmark(input_path, output_base + "/opening", "Opening", k, opening_parallel, opening_separable_parallel);
                csv_file << "Separable Evaluation," << scale << "," << k << "," << t << ",Opening," 
                         << r_ope.mean_t1 << "," << r_ope.ci_t1 << "," << r_ope.mean_t2 << "," << r_ope.ci_t2 << "," << r_ope.avg_speedup << "," << r_ope.ci_speedup << "\n";

                // --- TEST 4: CLOSING ---
                std::this_thread::sleep_for(std::chrono::seconds(2));
                auto r_clo = run_morphology_benchmark(input_path, output_base + "/closing", "Closing", k, closing_parallel, closing_separable_parallel);
                csv_file << "Separable Evaluation," << scale << "," << k << "," << t << ",Closing," 
                         << r_clo.mean_t1 << "," << r_clo.ci_t1 << "," << r_clo.mean_t2 << "," << r_clo.ci_t2 << "," << r_clo.avg_speedup << "," << r_clo.ci_speedup << "\n";
                
                csv_file.flush();
            }
        }
    }

    csv_file.close();
    std::cout << "\n[OK] Test Separable Parallelism completato con successo." << std::endl;
}

// Funzione che misura l'impatto ottenuto introducendo la vettorizzazione SIMD rispetto alle sole direttive di parallelismo OpenMP
void run_simd_impact_test(const std::vector<std::string>& scales, const std::vector<int>& thread_configs, const std::vector<int>& kernel_sizes) {
    std::string base_dir = "../results/simd_evaluation";
    if (!fs::exists(base_dir)) fs::create_directories(base_dir);

    std::string csv_path = base_dir + "/simd_impact_results.csv";
    std::ofstream csv_file(csv_path);
    csv_file << "Scale,KernelSize,Threads,TimeNoSIMD_Mean,TimeNoSIMD_CI,TimeSIMD_Mean,TimeSIMD_CI,SIMDSpeedup_Mean,SIMDSpeedup_CI\n";

    std::cout << "=== SIMD IMPACT TEST ===" << std::endl;
    std::cout << "CSV: " << csv_path << std::endl;

    for (const auto& scale : scales) {
        std::string input_path = "../data/dataset_grayscale/" + scale;
        std::string output_base = "../data/output/" + scale;

        if (!fs::exists(input_path)) {
            std::cerr << "[AVVISO] Cartella non trovata " << input_path << std::endl;
            continue;
        }

        for (int t : thread_configs) {
            omp_set_num_threads(t);
            for (int k : kernel_sizes) {
                std::cout << "\n>>> SIMD Test | Scale: " << scale << " | Threads: " << t << " | Kernel: " << k << "x" << k << " <<<" << std::endl;
                auto r = run_morphology_benchmark(input_path, output_base + "/eroded", "Impatto SIMD Kernel " + std::to_string(k) + " (" + std::to_string(t) + " Thread)", k, erode_parallel, erode_parallel_simd);
                
                csv_file << scale << "," << k << "," << t << ","
                         << r.mean_t1 << "," << r.ci_t1 << "," 
                         << r.mean_t2 << "," << r.ci_t2 << "," 
                         << r.avg_speedup << "," << r.ci_speedup << "\n";
                csv_file.flush();
            }
        }
    }
    csv_file.close();
    std::cout << "\n[OK] Test SIMD Impact completato con successo." << std::endl;
}

// Funzione che analizza la differenza di performance tra l'utilizzo della direttiva di scheduling statico e dinamico di OpenMP
void run_scheduling_impact_test(const std::vector<std::string>& scales, const std::vector<int>& thread_configs, const std::vector<int>& kernel_sizes) {
    std::string base_dir = "../results/scheduling_evaluation";
    if (!fs::exists(base_dir)) fs::create_directories(base_dir);

    std::string csv_path = base_dir + "/scheduling_impact_results.csv";
    std::ofstream csv_file(csv_path);
    csv_file << "Scale,KernelSize,Threads,TimeStatic_Mean,TimeStatic_CI,TimeDynamic_Mean,TimeDynamic_CI,DynamicVsStatic_Ratio\n";

    std::cout << "=== SCHEDULING IMPACT TEST ===" << std::endl;
    std::cout << "CSV: " << csv_path << std::endl;

    for (const auto& scale : scales) {
        std::string input_path = "../data/dataset_grayscale/" + scale;
        std::string output_base = "../data/output/" + scale;

        if (!fs::exists(input_path)) {
            std::cerr << "[AVVISO] Cartella non trovata " << input_path << std::endl;
            continue;
        }

        for (int t : thread_configs) {
            omp_set_num_threads(t);
            for (int k : kernel_sizes) {
                std::cout << "\n>>> Scheduling Test | Scale: " << scale << " | Threads: " << t << " | Kernel: " << k << "x" << k << " <<<" << std::endl;
                auto r = run_morphology_benchmark(input_path, output_base + "/eroded", 
                                                  "Scheduling (Static vs Dynamic) Kernel " + std::to_string(k) + " (" + std::to_string(t) + " Thread)", 
                                                  k, erode_parallel, erode_parallel_dynamic);
                
                csv_file << scale << "," << k << "," << t << ","
                         << r.mean_t1 << "," << r.ci_t1 << "," 
                         << r.mean_t2 << "," << r.ci_t2 << "," 
                         << r.avg_speedup << "\n";
                csv_file.flush();
            }
        }
    } 
    csv_file.close();
    std::cout << "\n[OK] Test Scheduling Impact completato con successo." << std::endl;
}

// Funzione per il test di accesso alla memoria (ROW VS COLUMN)
void run_memory_access_test(const std::vector<std::string>& scales, const std::vector<int>& thread_configs, const std::vector<int>& kernel_sizes) {
    std::string base_dir = "../results/memory_access_evaluation";
    if (!fs::exists(base_dir)) fs::create_directories(base_dir);

    std::string csv_path = base_dir + "/memory_access_results.csv";
    std::ofstream csv_file(csv_path);
    csv_file << "Scale,KernelSize,Threads,Operation,TimeRowMajor_Mean,TimeRowMajor_CI,TimeColumnMajor_Mean,TimeColumnMajor_CI,SpeedupRowVsCol_Mean,SpeedupRowVsCol_CI\n";

    std::cout << "=== MEMORY ACCESS TEST (ROW VS COLUMN) ===" << std::endl;
    std::cout << "CSV: " << csv_path << std::endl;

    for (const auto& scale : scales) {
        std::string input_path = "../data/dataset_grayscale/" + scale;
        std::string output_base = "../data/output/" + scale;

        if (!fs::exists(input_path)) {
            std::cerr << "[AVVISO] Cartella non trovata " << input_path << std::endl;
            continue;
        }

        for (int t : thread_configs) {
            omp_set_num_threads(t);
            for (int k : kernel_sizes) {
                std::cout << "\n>>> Memory Test | Scale: " << scale << " | Threads: " << t << " | Kernel: " << k << "x" << k << " <<<" << std::endl;
                
                std::this_thread::sleep_for(std::chrono::seconds(2));
                auto r_ero = run_morphology_benchmark(
                    input_path, output_base + "/eroded", 
                    "Erosione Memory Access", 
                    k, 
                    erode_parallel,          
                    erode_parallel_column
                );
                
                csv_file << scale << "," << k << "," << t << ",Erosione,"
                         << r_ero.mean_t1 << "," << r_ero.ci_t1 << "," 
                         << r_ero.mean_t2 << "," << r_ero.ci_t2 << "," 
                         << r_ero.avg_speedup << "," << r_ero.ci_speedup << "\n";

                std::this_thread::sleep_for(std::chrono::seconds(2));
                auto r_dil = run_morphology_benchmark(
                    input_path, output_base + "/dilated", 
                    "Dilatazione Memory Access", 
                    k, 
                    dilate_parallel,          
                    dilate_parallel_column   
                );
                
                csv_file << scale << "," << k << "," << t << ",Dilatazione,"
                         << r_dil.mean_t1 << "," << r_dil.ci_t1 << "," 
                         << r_dil.mean_t2 << "," << r_dil.ci_t2 << "," 
                         << r_dil.avg_speedup << "," << r_dil.ci_speedup << "\n";

                
                std::this_thread::sleep_for(std::chrono::seconds(2));
                auto r_ope = run_morphology_benchmark(
                    input_path, output_base + "/opening", 
                    "Opening Memory Access", 
                    k, 
                    opening_parallel,          
                    opening_parallel_column   
                );

                csv_file << scale << "," << k << "," << t << ",Opening,"
                         << r_ope.mean_t1 << "," << r_ope.ci_t1 << "," 
                         << r_ope.mean_t2 << "," << r_ope.ci_t2 << "," 
                         << r_ope.avg_speedup << "," << r_ope.ci_speedup << "\n";

                std::this_thread::sleep_for(std::chrono::seconds(2));
                auto r_clo = run_morphology_benchmark(
                    input_path, output_base + "/closing", 
                    "Closing Memory Access", 
                    k, 
                    closing_parallel,          
                    closing_parallel_column   
                );
                csv_file << scale << "," << k << "," << t << ",Closing,"
                         << r_clo.mean_t1 << "," << r_clo.ci_t1 << "," 
                         << r_clo.mean_t2 << "," << r_clo.ci_t2 << "," 
                         << r_clo.avg_speedup << "," << r_clo.ci_speedup << "\n";

                csv_file.flush();
            }
        }
    } 
    csv_file.close();
    std::cout << "\n[OK] Test Memory Access completato con successo." << std::endl;
}

// Funzione per il test di pipeline multi-thread, che confronta l'approccio tradizionale con una versione che sfrutta una pipeline di elaborazione con più thread
void run_pipeline_multithread_test(const std::vector<std::string>& scales, const std::vector<int>& thread_configs, const std::vector<int>& kernel_sizes) {    
    
    std::string base_dir = "../results/pipeline_multithread";
    if (!fs::exists(base_dir)) {
        fs::create_directories(base_dir);
    }

    std::string csv_path = base_dir + "/pipeline_multithread_results.csv";
    std::ofstream csv_file(csv_path);
    
    if (!csv_file.is_open()) {
        std::cerr << "[ERRORE] Impossibile creare il file CSV in: " << csv_path << std::endl;
        return;
    }
    
    csv_file << "Test,Scale,KernelSize,Threads,Operation,TimeStandard_Mean,TimeStandard_CI,TimePipelineMulti_Mean,TimePipelineMulti_CI,Speedup_Mean,Speedup_CI\n";

    std::cout << "=== PIPELINE MULTI-THREAD PARALLELISM TEST ===" << std::endl;
    std::cout << "CSV: " << csv_path << std::endl;

    for (const auto& scale : scales) {
        std::string input_path = "../data/dataset_grayscale/" + scale;
        std::string output_base = "../data/output/" + scale;

        if (!fs::exists(input_path)) {
            std::cerr << "[AVVISO] Cartella non trovata " << input_path << std::endl;
            continue;
        }

        for (int t : thread_configs) {
            std::cout << "\n>>> Configurazione: " << t << " Threads <<<" << std::endl;
            
            omp_set_num_threads(t);

            for (int k : kernel_sizes) {
                std::cout << "\n>>> Scale: " << scale << " | Threads: " << t << " | Kernel Size: " << k << "x" << k << " <<<" << std::endl;

                
                std::this_thread::sleep_for(std::chrono::seconds(2)); 
                
                auto r_ope = run_morphology_benchmark(input_path, output_base + "/opening", "Opening Multi-T", k, 
                                                      opening_parallel, 
                                                      opening_pipeline_multithread
                                                     );
                
                csv_file << "Pipeline Multi-Thread," << scale << "," << k << "," << t << ",Opening," 
                         << r_ope.mean_t1 << "," << r_ope.ci_t1 << "," << r_ope.mean_t2 << "," << r_ope.ci_t2 << "," << r_ope.avg_speedup << "," << r_ope.ci_speedup << "\n";

                std::this_thread::sleep_for(std::chrono::seconds(2));
                
                auto r_clo = run_morphology_benchmark(input_path, output_base + "/closing", "Closing Multi-T", k, 
                                                      closing_parallel, 
                                                      closing_pipeline_multithread
                                                     );
                
                csv_file << "Pipeline Multi-Thread," << scale << "," << k << "," << t << ",Closing," 
                         << r_clo.mean_t1 << "," << r_clo.ci_t1 << "," << r_clo.mean_t2 << "," << r_clo.ci_t2 << "," << r_clo.avg_speedup << "," << r_clo.ci_speedup << "\n";
                
                csv_file.flush(); 
            }
        }
    }

    csv_file.close();
    std::cout << "\n[OK] Test Pipeline Multi-Thread completato con successo." << std::endl;
}