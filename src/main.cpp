#include <vector>
#include "benchmark.h"

int main(int argc, char* argv[]) {

    //Strong scaling test
    std::string strong_scale = "scale_4.0x";
    std::vector<int> strong_scaling_threads = {1, 2, 4, 8, 16, 32};
    //run_strong_scaling_test(strong_scale, strong_scaling_threads);

    //Weak scaling test
    //run_weak_scaling_test();

    //Separable test
    std::vector<int> separable_threads = {1, 2, 4, 8};
    std::vector<std::string> separable_scales = {"scale_1.0x", "scale_2.0x"};
    std::vector<int> separable_kernel = {3, 7, 15};

    //run_separable_test(separable_scales, separable_threads, separable_kernel);


    //Scheduling impact test
    std::vector<int> scheduling_threads = {4, 8};
    std::vector<std::string> scheduling_scales = {"scale_2.0x"};
    std::vector<int> scheduling_kernel = {15};
    //run_scheduling_impact_test(scheduling_scales, scheduling_threads, scheduling_kernel);
    
    //SIMD impact test
    std::vector<int> simd_threads = {8};
    std::vector<std::string> simd_scales = {"scale_2.0x"};
    std::vector<int> simd_kernel = {15};
    //run_simd_impact_test(simd_scales, simd_threads, simd_kernel);

    //Memory access test
    std::vector<int> memory_threads = {1, 4};
    std::vector<std::string> memory_scales = {"scale_1.0x", "scale_4.0x"};
    std::vector<int> memory_kernel = {3};
    //run_memory_access_test(memory_scales, memory_threads, memory_kernel);

    //Pipeline multithread test
    std::vector<int> pipeline_mt_threads = {2, 4, 8};
    std::vector<std::string> pipeline_mt_scales = {"scale_1.0x", "scale_2.0x"};
    std::vector<int> pipeline_mt_kernel = {7};
    //run_pipeline_multithread_test(pipeline_mt_scales, pipeline_mt_threads, pipeline_mt_kernel);

    return 0;
}