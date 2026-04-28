#include "morphology.h"
#include <algorithm>

GrayImage erode_sequential(const GrayImage& input, int kernel_size) {
    int w = input.getWidth();
    int h = input.getHeight();
    GrayImage output(w, h);
    int offset = kernel_size / 2;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            unsigned char min_val = 255;

            for (int ky = -offset; ky <= offset; ++ky) {
                for (int kx = -offset; kx <= offset; ++kx) {
                    int nx = std::max(0, std::min(w - 1, x + kx));
                    int ny = std::max(0, std::min(h - 1, y + ky));

                    min_val = std::min(min_val, input.getPixel(nx, ny));
                    
                }
            }
            output.setPixel(x, y, min_val);
        }
    }
    return output;
}

GrayImage dilate_sequential(const GrayImage& input, int kernel_size) {
    int w = input.getWidth();
    int h = input.getHeight();
    GrayImage output(w, h);
    int offset = kernel_size / 2;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            unsigned char max_val = 0;

            for (int ky = -offset; ky <= offset; ++ky) {
                for (int kx = -offset; kx <= offset; ++kx) {
                    int nx = std::max(0, std::min(w - 1, x + kx));
                    int ny = std::max(0, std::min(h - 1, y + ky));

                    max_val = std::max(max_val, input.getPixel(nx, ny));
                }
            }
            output.setPixel(x, y, max_val);
        }
    }
    return output;
}

GrayImage opening_sequential(const GrayImage& input, int kernel_size) {
    GrayImage temp = erode_sequential(input, kernel_size);
    return dilate_sequential(temp, kernel_size);
}

GrayImage closing_sequential(const GrayImage& input, int kernel_size) {
    GrayImage temp = dilate_sequential(input, kernel_size);
    return erode_sequential(temp, kernel_size);
}