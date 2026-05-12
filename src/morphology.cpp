#include "morphology.h"
#include <algorithm>

GrayImage erode_sequential(const GrayImage& input, int kernel_size) {
    int w = input.getWidth();
    int h = input.getHeight();
    GrayImage output(w, h);
    int offset = kernel_size / 2;

    unsigned char min_val;
    int ky, kx, nx, ny;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            min_val = 255;

            for (ky = -offset; ky <= offset; ++ky) {
                for (kx = -offset; kx <= offset; ++kx) {
                    nx = std::max(0, std::min(w - 1, x + kx));
                    ny = std::max(0, std::min(h - 1, y + ky));

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

    unsigned char max_val;
    int ky, kx, nx, ny;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            max_val = 0;

            for (ky = -offset; ky <= offset; ++ky) {
                for (kx = -offset; kx <= offset; ++kx) {
                    nx = std::max(0, std::min(w - 1, x + kx));
                    ny = std::max(0, std::min(h - 1, y + ky));

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

GrayImage erode_parallel(const GrayImage& input, int kernel_size) {
    int w = input.getWidth();
    int h = input.getHeight();
    GrayImage output(w, h);
    int offset = kernel_size / 2;

    unsigned char min_val;
    int ky, kx, nx, ny;

    #pragma omp parallel for default(none) \
    shared(input, output, w, h, offset) \
    private(min_val, ky, kx, nx, ny) \
    schedule(static)
    // #pragma omp parallel for collapse(2)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            min_val = 255;

            for (ky = -offset; ky <= offset; ++ky) {
                #pragma omp simd reduction(min:min_val)
                for (kx = -offset; kx <= offset; ++kx) {
                    nx = std::max(0, std::min(w - 1, x + kx));
                    ny = std::max(0, std::min(h - 1, y + ky));

                    min_val = std::min(min_val, input.getPixel(nx, ny));
                    
                }
            }
            output.setPixel(x, y, min_val);
        }
    }
    return output;
}



GrayImage dilate_parallel(const GrayImage& input, int kernel_size) {
    int w = input.getWidth();
    int h = input.getHeight();
    GrayImage output(w, h);
    int offset = kernel_size / 2;

    unsigned char max_val;
    int ky, kx, nx, ny;

    #pragma omp parallel for default(none) \
    shared(input, output, w, h, offset) \
    private(max_val, ky, kx, nx, ny) \
    schedule(static)
    // #pragma omp parallel for collapse(2)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            max_val = 0;

            for (int ky = -offset; ky <= offset; ++ky) {
                #pragma omp simd reduction(max:max_val)
                for (kx = -offset; kx <= offset; ++kx) {
                    nx = std::max(0, std::min(w - 1, x + kx));
                    ny = std::max(0, std::min(h - 1, y + ky));

                    max_val = std::max(max_val, input.getPixel(nx, ny));
                }
            }
            output.setPixel(x, y, max_val);
        }
    }
    return output;
}

GrayImage opening_parallel(const GrayImage& input, int kernel_size) {
    GrayImage temp = erode_parallel(input, kernel_size);
    return dilate_parallel(temp, kernel_size);
}

GrayImage closing_parallel(const GrayImage& input, int kernel_size) {
    GrayImage temp = dilate_parallel(input, kernel_size);
    return erode_parallel(temp, kernel_size);
}