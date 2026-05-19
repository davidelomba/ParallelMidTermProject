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

GrayImage erode_parallel(const GrayImage& input, int kernel_size) {
    int w = input.getWidth();
    int h = input.getHeight();
    GrayImage output(w, h);
    int offset = kernel_size / 2;

    #pragma omp parallel for collapse(2) default(none) \
    shared(input, output, w, h, offset) \
    schedule(static)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            unsigned char min_val = 255;

            for (int ky = -offset; ky <= offset; ++ky) {
                //#pragma omp simd reduction(min:min_val)
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



GrayImage dilate_parallel(const GrayImage& input, int kernel_size) {
    int w = input.getWidth();
    int h = input.getHeight();
    GrayImage output(w, h);
    int offset = kernel_size / 2;


    #pragma omp parallel for collapse(2) default(none) \
    shared(input, output, w, h, offset) \
    schedule(static)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            unsigned char max_val = 0;

            for (int ky = -offset; ky <= offset; ++ky) {
                //#pragma omp simd reduction(max:max_val)
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

GrayImage opening_parallel(const GrayImage& input, int kernel_size) {
    GrayImage temp = erode_parallel(input, kernel_size);
    return dilate_parallel(temp, kernel_size);
}

GrayImage closing_parallel(const GrayImage& input, int kernel_size) {
    GrayImage temp = dilate_parallel(input, kernel_size);
    return erode_parallel(temp, kernel_size);
}

GrayImage erode_parallel_tiled(const GrayImage& input, int kernel_size) {
    int w = input.getWidth();
    int h = input.getHeight();
    GrayImage output(w, h);
    int offset = kernel_size / 2;

    const int TILE_SIZE = 32;

    #pragma omp parallel for collapse(2) default(none) \
    shared(input, output, w, h, offset) \
    schedule(static)
    for (int ty = 0; ty < h; ty += TILE_SIZE) {
        for (int tx = 0; tx < w; tx += TILE_SIZE) {

            for (int y = ty; y < std::min(ty + TILE_SIZE, h); ++y) {
                for (int x = tx; x < std::min(tx + TILE_SIZE, w); ++x) {
                    unsigned char min_val = 255;

                    for (int ky = -offset; ky <= offset; ++ky) {
                        //#pragma omp simd reduction(min:min_val)
                        for (int kx = -offset; kx <= offset; ++kx) {
                            int nx = std::max(0, std::min(w - 1, x + kx));
                            int ny = std::max(0, std::min(h - 1, y + ky));

                            min_val = std::min(min_val, input.getPixel(nx, ny));

                        }
                    }
                    output.setPixel(x, y, min_val);
                }
            }
        }
    }
    return output;
}

GrayImage dilate_parallel_tiled(const GrayImage& input, int kernel_size) {
    int w = input.getWidth();
    int h = input.getHeight();
    GrayImage output(w, h);
    int offset = kernel_size / 2;

    const int TILE_SIZE = 32;

    #pragma omp parallel for collapse(2) default(none) \
    shared(input, output, w, h, offset) \
    schedule(static)
    for (int ty = 0; ty < h; ty += TILE_SIZE) {
        for (int tx = 0; tx < w; tx += TILE_SIZE) {

            for (int y = ty; y < std::min(ty + TILE_SIZE, h); ++y) {
                for (int x = tx; x < std::min(tx + TILE_SIZE, w); ++x) {
                    unsigned char max_val = 0;

                    for (int ky = -offset; ky <= offset; ++ky) {
                        //#pragma omp simd reduction(max:max_val)
                        for (int kx = -offset; kx <= offset; ++kx) {
                            int nx = std::max(0, std::min(w - 1, x + kx));
                            int ny = std::max(0, std::min(h - 1, y + ky));

                            max_val = std::max(max_val, input.getPixel(nx, ny));
                        }
                    }
                    output.setPixel(x, y, max_val);
                }
            }
        }
    }
    return output;
}

GrayImage opening_parallel_tiled(const GrayImage& input, int kernel_size) {
    GrayImage temp = erode_parallel_tiled(input, kernel_size);
    return dilate_parallel_tiled(temp, kernel_size);
}

GrayImage closing_parallel_tiled(const GrayImage& input, int kernel_size) {
    GrayImage temp = dilate_parallel_tiled(input, kernel_size);
    return erode_parallel_tiled(temp, kernel_size);
}

GrayImage erode_separable_parallel(const GrayImage& input, int kernel_size) {
    int w = input.getWidth();
    int h = input.getHeight();
    int offset = kernel_size / 2;
    
    GrayImage temp(w, h);
    GrayImage output(w, h);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            unsigned char min_val = 255;

            #pragma omp simd reduction(min:min_val)
            for (int kx = -offset; kx <= offset; ++kx) {
                int nx = std::max(0, std::min(w - 1, x + kx));
                min_val = std::min(min_val, input.getPixel(nx, y));
            }
            temp.setPixel(x, y, min_val);
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            unsigned char min_val = 255;
            
            //#pragma omp simd reduction(min:min_val)
            for (int ky = -offset; ky <= offset; ++ky) {
                int ny = std::max(0, std::min(h - 1, y + ky));
                min_val = std::min(min_val, temp.getPixel(x, ny));
            }
            output.setPixel(x, y, min_val);
        }
    }

    return output;
}

GrayImage dilate_separable_parallel(const GrayImage& input, int kernel_size) {
    int w = input.getWidth();
    int h = input.getHeight();
    int offset = kernel_size / 2;
    
    GrayImage temp(w, h);
    GrayImage output(w, h);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            unsigned char max_val = 0;
            
            //#pragma omp simd reduction(max:max_val)
            for (int kx = -offset; kx <= offset; ++kx) {
                int nx = std::max(0, std::min(w - 1, x + kx));
                max_val = std::max(max_val, input.getPixel(nx, y));
            }
            temp.setPixel(x, y, max_val);
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            unsigned char max_val = 0;
            
            for (int ky = -offset; ky <= offset; ++ky) {
                int ny = std::max(0, std::min(h - 1, y + ky));
                max_val = std::max(max_val, temp.getPixel(x, ny));
            }
            output.setPixel(x, y, max_val);
        }
    }

    return output;
}

GrayImage opening_separable_parallel(const GrayImage& input, int kernel_size) {
    GrayImage temp = erode_separable_parallel(input, kernel_size);
    return dilate_separable_parallel(temp, kernel_size);
}

GrayImage closing_separable_parallel(const GrayImage& input, int kernel_size) {
    GrayImage temp = dilate_separable_parallel(input, kernel_size);
    return erode_separable_parallel(temp, kernel_size);
}


void compute_erosion_for_row(const GrayImage& input, GrayImage& output, int y, int kernel_size) {
    int width = input.getWidth();
    int height = input.getHeight();
    int radius = kernel_size / 2;

    for (int x = 0; x < width; ++x) {
        unsigned char min_val = 255; 

        for (int ky = -radius; ky <= radius; ++ky) {
            int ny = std::max(0, std::min(y + ky, height - 1));
            
            for (int kx = -radius; kx <= radius; ++kx) {
                int nx = std::max(0, std::min(x + kx, width - 1));

                unsigned char pixel = input.getPixel(nx, ny); 
                if (pixel < min_val) {
                    min_val = pixel;
                }
            }
        }
        output.setPixel(x, y, min_val); 
    }
}

void compute_dilation_for_row(const GrayImage& input, GrayImage& output, int y, int kernel_size) {
    int width = input.getWidth();
    int height = input.getHeight();
    int radius = kernel_size / 2;

    for (int x = 0; x < width; ++x) {
        unsigned char max_val = 0; 

        for (int ky = -radius; ky <= radius; ++ky) {
            int ny = std::max(0, std::min(y + ky, height - 1));

            for (int kx = -radius; kx <= radius; ++kx) {
                int nx = std::max(0, std::min(x + kx, width - 1));

                unsigned char pixel = input.getPixel(nx, ny); 
                if (pixel > max_val) {
                    max_val = pixel;
                }
            }
        }
        output.setPixel(x, y, max_val); 
    }
}


GrayImage opening_pipeline(const GrayImage& input, int kernel_size) {
    int radius = kernel_size / 2;
    int width = input.getWidth();
    int height = input.getHeight();
    
    GrayImage temp_eroded(width, height);
    GrayImage output(width, height);
    
    int max_ready_row = -1; 
    const int chunk_size = 32; 

    #pragma omp parallel num_threads(2) shared(max_ready_row, temp_eroded, output)
    {
        #pragma omp sections
        {
            // THREAD 0: PRODUCER (EROSIONE)
            #pragma omp section
            {
                for (int y = 0; y < height; ++y) {
                    compute_erosion_for_row(input, temp_eroded, y, kernel_size);
                    
                    if ((y + 1) % chunk_size == 0 || y == height - 1) {
                        #pragma omp atomic write
                        max_ready_row = y;
                    }
                }
            }

            // THREAD 1: CONSUMER (DILATAZIONE)
            #pragma omp section
            {
                for (int y = 0; y < height; ++y) {
                    int needed_row = std::min(y + radius, height - 1);
                    int current_ready;

                    do {
                        #pragma omp atomic read
                        current_ready = max_ready_row;
                    } while (current_ready < needed_row);

                    compute_dilation_for_row(temp_eroded, output, y, kernel_size);
                }
            }
        } 
    } 
    
    return output;
}

GrayImage closing_pipeline(const GrayImage& input, int kernel_size) {
    int radius = kernel_size / 2;
    int width = input.getWidth();
    int height = input.getHeight();
    
    GrayImage temp_dilated(width, height);
    GrayImage output(width, height);
    
    int max_ready_row = -1; 
    const int chunk_size = 32; 

    #pragma omp parallel num_threads(2) shared(max_ready_row, temp_dilated, output)
    {
        #pragma omp sections
        {
            // THREAD 0: PRODUCER (DILATAZIONE)
            #pragma omp section
            {
                for (int y = 0; y < height; ++y) {
                    compute_dilation_for_row(input, temp_dilated, y, kernel_size);
                    
                    if ((y + 1) % chunk_size == 0 || y == height - 1) {
                        #pragma omp atomic write
                        max_ready_row = y;
                    }
                }
            }

            // THREAD 1: CONSUMER (EROSIONE)
            #pragma omp section
            {
                for (int y = 0; y < height; ++y) {
                    int needed_row = std::min(y + radius, height - 1);
                    int current_ready;

                    do {
                        #pragma omp atomic read
                        current_ready = max_ready_row;
                    } while (current_ready < needed_row);

                    compute_erosion_for_row(temp_dilated, output, y, kernel_size);
                }
            }
        } 
    } 
    
    return output;
}