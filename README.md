# Grayscale Image Morphology Benchmark

This repository contains a high-performance **C++17** framework designed to benchmark morphological operations (Erosion, Dilation, Opening, Closing) on grayscale images from Berkeley Segmentation Dataset 500 (BSDS500). It compares a standard sequential implementation against an optimized, parallelized versions using **OpenMP**. 

The framework evaluates several optimization strategies, including 1D separable kernels, SIMD vectorization, memory access patterns (row-major vs. column-major), scheduling policies (static vs. dynamic), and pipeline multithreading for compound operations. It also includes rigorous statistical validation using Student's T-distribution for 95% confidence intervals and automated Python tools for dataset preparation and performance plotting.

## 🚀 Features

- **Morphological Operations**: Complete implementations of Grayscale Erosion, Dilation, Opening, and Closing.
- **Advanced Parallelization & Optimizations (OpenMP)**:
  - **1D Separable Kernels**: Reduces computational complexity from O(N × K²) to O(N × 2K) where K is the kernel size.
  - **SIMD Vectorization**: Leverages hardware-level vector registers via `#pragma omp simd`.
  - **Memory Access Patterns**: Designed around Row-Major layout to maximize CPU cache locality.
  - **Thread Scheduling**: Evaluates and compares `static` and `dynamic` OpenMP scheduling policies.
  - **Pipeline Multithreading**: Parallelizes compound operations (Opening/Closing) by pipelining sequential steps.
- **Rigorous Statistical Benchmarking**:
  - **Warm-up Phase**: Wakes up the OpenMP thread pool and populates caches before timing.
  - **Cold Cache per Run**: Reloads images from disk on each iteration to accurately simulate real-world I/O overhead.
  - **High Accuracy**: Computes the mean, standard deviation, and **95% Confidence Interval** using Student's T-distribution critical values over 5 independent runs.
  - **Automatic Metrics**: Computes Speedup and Efficiency values.
  - **Correctness Validation**: Strictly validates parallel pixel outputs against the sequential baseline to guarantee 100% mathematical accuracy.
- **Automated Data & Visualization Pipeline**:
  - Scripts to fetch, convert, and multi-scale the BSDS500 dataset.
  - Automated graph plotting tools for analysis using Pandas, Matplotlib, and Seaborn.

## 📂 Project Structure

```text
├── CMakeLists.txt             # CMake build configuration
├── requirements.txt           # Python environment dependencies
├── include/                   # Header files (.h)
│   ├── GrayImage.h            # Wrapper class for 1D-linearized image memory
│   ├── benchmark.h            # Benchmarking framework and statistical structures
│   ├── morphology.h           # Sequential and parallel morphological operations
│   ├── stb_image.h            # Single-header image loading library
│   └── stb_image_write.h      # Single-header image saving library
├── src/                       # Source files (.cpp)
│   ├── main.cpp               # System entry point
│   ├── GrayImage.cpp          # Image management and I/O integration
│   ├── morphology.cpp         # Implementation of morphology algorithms & OpenMP directives
│   └── benchmark.cpp          # Benchmarking tests, validation, and CSV export logic
├── script/                    # Python automation, dataset prep, and plot generation scripts
│   ├── download_dataset.py    # Downloads the master BSDS500 image dataset
│   ├── convert_to_grayscale.py    # Pre-processes images into 8-bit grayscale
│   ├── scale_grayscale_dataset.py # Generates resolution variants (1.0x, 2.0x, 4.0x)
│   ├── strong_scaling.py      # Generates plots for strong scaling test
│   ├── weak_scaling.py        # Generates plots for weak scaling test
│   ├── separable.py           # Generates plots for separbale kernel test
│   ├── pipeline_multithread.py    # Generates plots for pipeline test
│   └── optimal_vs_sequential.py   # Generates plots for optimal vs sequential test
├── results/                   # Test results directory (generated automatically by scripts)
├── results/                   # Graphs directory (generated automatically by scripts)
└── data/                      # Data directory (generated automatically by scripts)
```

## 🛠️ Prerequisites

- **C++ Compiler**: GCC, Clang, or MSVC with full **C++17** support.
- **CMake**: Version 3.10 or higher.
- **OpenMP**: Required for thread-level and SIMD parallelization.
- **Python 3.x**: Required for dataset setup and data visualization.

---

## ⚙️ Setup and Execution

### 1. Python Environment & Data Preparation
First, set up your Python environment and prepare the benchmark dataset. Run these commands from the project root directory:

```bash
# Install all required Python dependencies (Pillow, Pandas, Matplotlib, Seaborn)
pip install -r requirements.txt

# Execute data management pipeline
python script/download_dataset.py
python script/convert_to_grayscale.py
python script/scale_grayscale_dataset.py
```
This automatically downloads the BSDS500 dataset and sets up the multi-scale grayscale images under the `data/` directory.

### 2. Compile the C++ Framework
Configure and build the native benchmarking executable using CMake:

```bash
# Create and navigate to build directory
mkdir build
cd build

# Generate build files and compile (Release mode recommended for true benchmarking)
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

### 3. Run the Benchmark
Execute the compiled binary to start testing. The framework will benchmark the dataset and export the data points into a structured CSV file:

```bash
# Run from inside the build folder or project root depending on target location
./main
```

### 4. Generate Performance Visualization Plots
Once the benchmark finishes the CSV results can be seen in `results/`. Morover, you can visualize the scaling and speedup properties using the plot generation scripts provided in the `script/` folder. Running these scripts will parse the CSV data and automatically save the resulting charts as PNG images.
