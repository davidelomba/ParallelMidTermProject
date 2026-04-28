#pragma once
#include "GrayImage.h"
GrayImage erode_sequential(const GrayImage& input, int kernel_size);
GrayImage dilate_sequential(const GrayImage& input, int kernel_size);