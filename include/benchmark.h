#include <string>
#include <functional>
#include "GrayImage.h"

using MorphoFunc = std::function<GrayImage(const GrayImage&, int)>;

void run_morphology_benchmark(
    const std::string& input_path,
    const std::string& output_path,
    const std::string& label,
    int kernel_size,
    MorphoFunc seq_func,
    MorphoFunc par_func
);