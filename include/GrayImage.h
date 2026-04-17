#pragma once
#include <vector>
#include <string>

class GrayImage {
private:
    int width;
    int height;
    std::vector<unsigned char> data;

public:
    GrayImage() : width(0), height(0) {}
    GrayImage(int w, int h);
    
    bool load(const std::string& path);
    bool save(const std::string& path) const;

    int getWidth() const { return width; }
    int getHeight() const { return height; }
    
    unsigned char getPixel(int x, int y) const {
        return data[y * width + x];
    }
    
    void setPixel(int x, int y, unsigned char value) {
        data[y * width + x] = value;
    }

    unsigned char* getRawData() { return data.data(); }
    const unsigned char* getRawData() const { return data.data(); }
};