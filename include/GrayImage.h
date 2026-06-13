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
    
    // Legge il valore di un pixel alle coordinate (x, y) (y sono le righe, x sono le colonne a partire dall'angolo in alto a sinistra)
    unsigned char getPixel(int x, int y) const {
        return data[y * width + x];
    }
    
    // Modifica il valore di un pixel alle coordinate (x, y)
    void setPixel(int x, int y, unsigned char value) {
        data[y * width + x] = value;
    }

    unsigned char* getRawData() { return data.data(); }
    const unsigned char* getRawData() const { return data.data(); }
};