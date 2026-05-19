#pragma once
#include "GrayImage.h"
GrayImage erode_sequential(const GrayImage& input, int kernel_size);
GrayImage dilate_sequential(const GrayImage& input, int kernel_size);
GrayImage opening_sequential(const GrayImage& input, int kernel_size);
GrayImage closing_sequential(const GrayImage& input, int kernel_size);

GrayImage erode_parallel(const GrayImage& input, int kernel_size);
GrayImage dilate_parallel(const GrayImage& input, int kernel_size);
GrayImage opening_parallel(const GrayImage& input, int kernel_size);
GrayImage closing_parallel(const GrayImage& input, int kernel_size);

GrayImage erode_parallel_tiled(const GrayImage& input, int kernel_size);
GrayImage dilate_parallel_tiled(const GrayImage& input, int kernel_size);
GrayImage opening_parallel_tiled(const GrayImage& input, int kernel_size);
GrayImage closing_parallel_tiled(const GrayImage& input, int kernel_size);

GrayImage erode_separable_parallel(const GrayImage& input, int kernel_size);
GrayImage dilate_separable_parallel(const GrayImage& input, int kernel_size);
GrayImage opening_separable_parallel(const GrayImage& input, int kernel_size);
GrayImage closing_separable_parallel(const GrayImage& input, int kernel_size);