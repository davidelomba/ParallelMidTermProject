#include <string>
#include <functional>
#include "GrayImage.h"

struct BenchResult {
    double mean_t1;
    double ci_t1;
    double mean_t2;
    double ci_t2;
    double avg_speedup;
    double ci_speedup;
};

// Definizione dell'alias di tipo per le funzioni morfologiche
using MorphoFunc = std::function<GrayImage(const GrayImage&, int)>;

BenchResult run_morphology_benchmark(
    const std::string& input_path,
    const std::string& output_path,
    const std::string& label,
    int kernel_size,
    MorphoFunc func1,
    MorphoFunc func2
);

void save_benchmark_to_csv(std::ofstream& file, std::string law_name, std::string scale, int threads, std::string op, BenchResult res);

void run_strong_scaling_test(const std::string& scale, const std::vector<int>& thread_configs);

void run_weak_scaling_test();

void run_separable_test(const std::vector<std::string>& scales, const std::vector<int>& thread_configs, const std::vector<int>& kernel_sizes);

void run_simd_impact_test(const std::vector<std::string>& scales, const std::vector<int>& thread_configs, const std::vector<int>& kernel_sizes);

void run_scheduling_impact_test(const std::vector<std::string>& scales, const std::vector<int>& thread_configs, const std::vector<int>& kernel_sizes);

void run_memory_access_test(const std::vector<std::string>& scales, const std::vector<int>& thread_configs, const std::vector<int>& kernel_sizes);

void run_pipeline_multithread_test(const std::vector<std::string>& scales, const std::vector<int>& thread_configs, const std::vector<int>& kernel_sizes);

void run_optimal_vs_sequential_test(const std::vector<std::string>& scales, const std::vector<int>& thread_configs, const std::vector<int>& kernel_sizes);
