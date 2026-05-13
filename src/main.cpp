#include <vector>
#include "benchmark.h"

int main(int argc, char* argv[]) {
    std::string scale_to_test = "scale_2.0x";
    std::vector<int> my_threads = {1, 2, 4, 8, 16, 32};

    run_strong_scaling_test(scale_to_test, my_threads);
    run_weak_scaling_test();

    return 0;
}