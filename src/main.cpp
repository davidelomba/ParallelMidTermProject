#include <vector>
#include "benchmark.h"

int main(int argc, char* argv[]) {
    std::string scale_to_test = "scale_2.0x";
    std::vector<int> my_threads = {1, 2, 4, 8, 16, 32};
    int max_threads = 8;


    run_strong_scaling_test(scale_to_test, my_threads);
    run_weak_scaling_test();

    std::vector<int> kernel_config = {3, 7, 11, 15};

    run_separable_test({scale_to_test}, max_threads, kernel_config);
    run_pipeline_test({scale_to_test}, kernel_config);
    return 0;
}