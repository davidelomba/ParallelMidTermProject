#include "GrayImage.h"


GrayImage erode_sequential(const GrayImage& input, int kernel_size);
GrayImage dilate_sequential(const GrayImage& input, int kernel_size);
GrayImage opening_sequential(const GrayImage& input, int kernel_size);
GrayImage closing_sequential(const GrayImage& input, int kernel_size);

GrayImage erode_parallel(const GrayImage& input, int kernel_size);
GrayImage dilate_parallel(const GrayImage& input, int kernel_size);
GrayImage opening_parallel(const GrayImage& input, int kernel_size);
GrayImage closing_parallel(const GrayImage& input, int kernel_size);

GrayImage erode_separable_parallel(const GrayImage& input, int kernel_size);
GrayImage dilate_separable_parallel(const GrayImage& input, int kernel_size);
GrayImage opening_separable_parallel(const GrayImage& input, int kernel_size);
GrayImage closing_separable_parallel(const GrayImage& input, int kernel_size);

GrayImage erode_parallel_simd(const GrayImage& input, int kernel_size);
GrayImage erode_parallel_dynamic(const GrayImage& input, int kernel_size);

GrayImage erode_parallel_column(const GrayImage& input, int kernel_size);
GrayImage dilate_parallel_column(const GrayImage& input, int kernel_size);
GrayImage opening_parallel_column(const GrayImage& input, int kernel_size);
GrayImage closing_parallel_column(const GrayImage& input, int kernel_size);

GrayImage opening_pipeline_multithread(const GrayImage& input, int kernel_size);
GrayImage closing_pipeline_multithread(const GrayImage& input, int kernel_size);
