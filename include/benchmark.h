#include <string>
#include <functional>
#include "GrayImage.h"

struct BenchResult {
    double total_t_seq;
    double total_t_par;
    double avg_speedup;
};

using MorphoFunc = std::function<GrayImage(const GrayImage&, int)>;

BenchResult run_morphology_benchmark(
    const std::string& input_path,
    const std::string& output_path,
    const std::string& label,
    int kernel_size,
    MorphoFunc seq_func,
    MorphoFunc par_func
);

void save_benchmark_to_csv(std::ofstream& file, std::string law_name, std::string scale, int threads, std::string op, BenchResult res);

void run_strong_scaling_test(const std::string& scale, const std::vector<int>& thread_configs);

void run_weak_scaling_test();
