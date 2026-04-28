#include "GrayImage.h"
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

GrayImage::GrayImage(int w, int h) : width(w), height(h), data(w * h, 0) {}

bool GrayImage::load(const std::string& path) {
    int w, h, channels;
    unsigned char* img = stbi_load(path.c_str(), &w, &h, &channels, 1);
    
    if (!img) {
        std::cerr << "Fallito caricamento: " << path << std::endl;
        return false;
    }

    width = w;
    height = h;
    
    data.assign(img, img + (w * h));
    
    stbi_image_free(img);
    return true;
}

bool GrayImage::save(const std::string& path) const {
    if (data.empty()) return false;
    return stbi_write_png(path.c_str(), width, height, 1, data.data(), width) != 0;
}