#include "morphology.h"
#include <algorithm>
#include <omp.h>
#include <vector>
#include <iostream>

// Funzione che esegue l'erosione morfologica sequenziale su un'immagine in scala di grigi utilizzando un kernel quadrato
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

// Funzione che esegue la dilatazione morfologica sequenziale su un'immagine in scala di grigi utilizzando un kernel quadrato
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

// Funzione che esegue l'operazione di opening morfologica sequenziale su un'immagine in scala di grigi utilizzando un kernel quadrato
GrayImage opening_sequential(const GrayImage& input, int kernel_size) {
    GrayImage temp = erode_sequential(input, kernel_size);
    return dilate_sequential(temp, kernel_size);
}

// Funzione che esegue l'operazione di closing morfologica sequenziale su un'immagine in scala di grigi utilizzando un kernel quadrato
GrayImage closing_sequential(const GrayImage& input, int kernel_size) {
    GrayImage temp = dilate_sequential(input, kernel_size);
    return erode_sequential(temp, kernel_size);
}

// Funzione che esegue l'erosione morfologica in parallelo su un'immagine in scala di grigi collassando i cicli spaziali e usando lo scheduling statico
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

// Funzione che esegue la dilatazione morfologica in parallelo su un'immagine in scala di grigi collassando i cicli spaziali e usando lo scheduling statico
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

// Funzione che esegue l'operazione di opening morfologica in parallelo su un'immagine in scala di grigi sfruttando le funzioni parallele di erosione e dilatazione
GrayImage opening_parallel(const GrayImage& input, int kernel_size) {
    GrayImage temp = erode_parallel(input, kernel_size);
    return dilate_parallel(temp, kernel_size);
}

// Funzione che esegue l'operazione di closing morfologica in parallelo su un'immagine in scala di grigi sfruttando le funzioni parallele di erosione e dilatazione
GrayImage closing_parallel(const GrayImage& input, int kernel_size) {
    GrayImage temp = dilate_parallel(input, kernel_size);
    return erode_parallel(temp, kernel_size);
}

// Funzione che ottimizza l'erosione parallela separando il kernel bidimensionale in due passaggi unidimensionali (orizzontale e poi verticale)
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

            //#pragma omp simd reduction(min:min_val)
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
            
            for (int ky = -offset; ky <= offset; ++ky) {
                int ny = std::max(0, std::min(h - 1, y + ky));
                min_val = std::min(min_val, temp.getPixel(x, ny));
            }
            output.setPixel(x, y, min_val);
        }
    }

    return output;
}

// Funzione che ottimizza la dilatazione parallela separando il kernel bidimensionale in due passaggi unidimensionali (orizzontale e poi verticale)
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

// Funzione che esegue l'operazione di opening morfologica in parallelo su un'immagine in scala di grigi sfruttando le funzioni parallele separabili di erosione e dilatazione
GrayImage opening_separable_parallel(const GrayImage& input, int kernel_size) {
    GrayImage temp = erode_separable_parallel(input, kernel_size);
    return dilate_separable_parallel(temp, kernel_size);
}

// Funzione che esegue l'operazione di closing morfologica in parallelo su un'immagine in scala di grigi sfruttando le funzioni parallele separabili di erosione e dilatazione
GrayImage closing_separable_parallel(const GrayImage& input, int kernel_size) {
    GrayImage temp = dilate_separable_parallel(input, kernel_size);
    return erode_separable_parallel(temp, kernel_size);
}

// Funzione ausiliaria che calcola localmente l'erosione morfologica per una singola riga specifica 'y' dell'immagine
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

// Funzione ausiliaria che calcola localmente la dilatazione morfologica per una singola riga specifica 'y' dell'immagine
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

// Funzione che esegue l'erosione parallela applicando anche la vettorizzazione SIMD sul ciclo di scansione orizzontale del kernel
GrayImage erode_parallel_simd(const GrayImage& input, int kernel_size) {
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
                #pragma omp simd reduction(min:min_val)
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

// Funzione che esegue l'erosione parallela con scheduling dinamico, senza vettorizzazione SIMD
GrayImage erode_parallel_dynamic(const GrayImage& input, int kernel_size) {
    int w = input.getWidth();
    int h = input.getHeight();
    GrayImage output(w, h);
    int offset = kernel_size / 2;

    #pragma omp parallel for collapse(2) default(none) \
    shared(input, output, w, h, offset) \
    schedule(dynamic, 16)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            unsigned char min_val = 255;

            for (int ky = -offset; ky <= offset; ++ky) {
                // #pragma omp simd reduction(min:min_val)
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

// Funzione che esegue l'erosione parallela applicando la scansione per colonne invece che per righe, per valutare l'impatto dell'accesso alla memoria
GrayImage erode_parallel_column(const GrayImage& input, int kernel_size) {
    int w = input.getWidth();
    int h = input.getHeight();
    GrayImage output(w, h);
    int offset = kernel_size / 2;

    #pragma omp parallel for collapse(2) default(none) \
    shared(input, output, w, h, offset) \
    schedule(static)
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            unsigned char min_val = 255;
            // #pragma omp simd reduction(min:min_val)
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

// Funzione che esegue la dilatazione parallela applicando la scansione per colonne invece che per righe, per valutare l'impatto dell'accesso alla memoria
GrayImage dilate_parallel_column(const GrayImage& input, int kernel_size) {
    int w = input.getWidth();
    int h = input.getHeight();
    GrayImage output(w, h);
    int offset = kernel_size / 2;

    #pragma omp parallel for collapse(2) default(none) \
    shared(input, output, w, h, offset) \
    schedule(static)
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            unsigned char max_val = 0;

            for (int ky = -offset; ky <= offset; ++ky) {
                // #pragma omp simd reduction(max:max_val)
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

// Funzione che esegue l'operazione di opening morfologica in parallelo su un'immagine in scala di grigi sfruttando le funzioni parallele di erosione e dilatazione con accesso per colonne
GrayImage opening_parallel_column(const GrayImage& input, int kernel_size) {
    GrayImage temp = erode_parallel_column(input, kernel_size);
    return dilate_parallel_column(temp, kernel_size);
}

// Funzione che esegue l'operazione di closing morfologica in parallelo su un'immagine in scala di grigi sfruttando le funzioni parallele di erosione e dilatazione con accesso per colonne
GrayImage closing_parallel_column(const GrayImage& input, int kernel_size) {
    GrayImage temp = dilate_parallel_column(input, kernel_size);
    return erode_parallel_column(temp, kernel_size);
}

// Funzione che implementa l'operazione di opening tramite un modello di pipeline concorrente dividendo i thread in producers (erosione) e consumers (dilatazione)
GrayImage opening_pipeline_multithread(const GrayImage& input, int kernel_size) {
    int num_threads = omp_get_max_threads(); 
    
    // La pipeline richiede almeno 2 thread (1 prod e 1 cons)
    if (num_threads < 2) {
        num_threads = 2; 
    }

    int radius = kernel_size / 2;
    int width = input.getWidth();
    int height = input.getHeight();

    GrayImage output(width, height);
    
    // Alloca solo l'immagine temporanea intermedia
    GrayImage temp_eroded(width, height);
    
    
    const int chunk_size = 32; // Dimensione del blocco di righe
    int num_chunks = (height + chunk_size - 1) / chunk_size;
    
    // Array di sincronizzazione: chunk_ready[c] = 1 significa che il chunk 'c' è pronto
    std::vector<int> chunk_ready(num_chunks, 0);

    #pragma omp parallel num_threads(num_threads) shared(chunk_ready, temp_eroded, output, input)
    {
        int my_id = omp_get_thread_num();
        int total_threads = omp_get_num_threads();
        
        // Divide i thread a metà
        int prod_threads = total_threads / 2;
        int cons_threads = total_threads - prod_threads;
        
        if (my_id < prod_threads) {
            // PRODUCER: EROSIONE
            int prod_id = my_id; // ID locale al gruppo (da 0 a prod_threads-1)
            
            // Distribuzione ciclica dei chunk tra i produttori
            for (int c = prod_id; c < num_chunks; c += prod_threads) {
                int start_y = c * chunk_size;
                int end_y = std::min(height, start_y + chunk_size);
                
                // Elaborazione di tutte le righe assegnate a questo chunk
                for (int y = start_y; y < end_y; ++y) {
                    compute_erosion_for_row(input, temp_eroded, y, kernel_size);
                }
                
                // Garanzia di visibilità: assicura che tutte le scritture su 'temp_eroded' siano visibili prima di notificare che il chunk è pronto
                #pragma omp flush(temp_eroded) 
                // Notifica atomica: il chunk 'c' è interamente completato
                #pragma omp atomic write
                chunk_ready[c] = 1;

                // Garanzia di visibilità: assicura che la scrittura su 'chunk_ready' sia visibile a tutti i consumatori
                #pragma omp flush(chunk_ready)
            }
        } 
        else {
            // CONSUMER: DILATAZIONE
            int cons_id = my_id - prod_threads; // ID locale al gruppo (da 0 a cons_threads-1)
            
            // Distribuzione ciclica dei chunk tra i consumatori
            for (int c = cons_id; c < num_chunks; c += cons_threads) {
                int start_y = c * chunk_size;
                int end_y = std::min(height, start_y + chunk_size);
                
                // Calcolo della riga minima e massima necessarie per calcolare questo intero chunk
                int min_needed_row = std::max(0, start_y - radius);
                int max_needed_row = std::min(height - 1, end_y - 1 + radius);
                
                // Conversione delle righe necessarie nei rispettivi indici di chunk
                int min_needed_chunk = min_needed_row / chunk_size;
                int max_needed_chunk = max_needed_row / chunk_size;
                
                // Attesa attiva (Spin-Wait) mirata solo sui chunk strettamente necessari
                for (int nc = min_needed_chunk; nc <= max_needed_chunk; ++nc) {
                    int ready = 0;
                    while (!ready) {
                        #pragma omp atomic read
                        ready = chunk_ready[nc];

                    }

                }

                // Garanzia di visibilità: assicura che tutte le scritture su 'chunk_ready' siano visibili prima di iniziare la dilatazione
                #pragma omp flush(chunk_ready)

                // Garanzia di visibilità: assicura che tutte le scritture su 'temp_eroded' siano visibili prima di iniziare la dilatazione
                #pragma omp flush(temp_eroded)
                
                // Una volta che tutti i chunk necessari sono pronti, viene elaborato il blocco scrivendo su 'output'
                for (int y = start_y; y < end_y; ++y) {
                    compute_dilation_for_row(temp_eroded, output, y, kernel_size);
                }
            }
        }
    }
    return output;
}

// Funzione che implementa l'operazione di closing tramite un modello di pipeline concorrente dividendo i thread in producers (dilatazione) e consumers (erosione)
GrayImage closing_pipeline_multithread(const GrayImage& input, int kernel_size) {
    int num_threads = omp_get_max_threads(); 
    if (num_threads < 2) {
        num_threads = 2; 
    }

    int radius = kernel_size / 2;
    int width = input.getWidth();
    int height = input.getHeight();

    GrayImage output(width, height);
    
    // Immagine temporanea per la dilatazione
    GrayImage temp_dilated(width, height);
    
    const int chunk_size = 32; 
    int num_chunks = (height + chunk_size - 1) / chunk_size;
    std::vector<int> chunk_ready(num_chunks, 0);

    #pragma omp parallel num_threads(num_threads) shared(chunk_ready, temp_dilated, output, input)
    {
        int my_id = omp_get_thread_num();
        int total_threads = omp_get_num_threads();
        
        int prod_threads = total_threads / 2;
        int cons_threads = total_threads - prod_threads;
        
        if (my_id < prod_threads) {
            // PRODUCER: DILATAZIONE
            int prod_id = my_id;
            for (int c = prod_id; c < num_chunks; c += prod_threads) {
                int start_y = c * chunk_size;
                int end_y = std::min(height, start_y + chunk_size);
                
                for (int y = start_y; y < end_y; ++y) {
                    compute_dilation_for_row(input, temp_dilated, y, kernel_size);
                }

                #pragma omp flush(temp_dilated) 
                
                #pragma omp atomic write
                chunk_ready[c] = 1;

                #pragma omp flush(chunk_ready)
            }
        } 
        else {
            // CONSUMER: EROSIONE
            int cons_id = my_id - prod_threads;
            for (int c = cons_id; c < num_chunks; c += cons_threads) {
                int start_y = c * chunk_size;
                int end_y = std::min(height, start_y + chunk_size);
                
                int min_needed_row = std::max(0, start_y - radius);
                int max_needed_row = std::min(height - 1, end_y - 1 + radius);
                
                int min_needed_chunk = min_needed_row / chunk_size;
                int max_needed_chunk = max_needed_row / chunk_size;
                
                for (int nc = min_needed_chunk; nc <= max_needed_chunk; ++nc) {
                    int ready = 0;
                    while (!ready) {
                        #pragma omp atomic read
                        ready = chunk_ready[nc];
                    }
                }

                #pragma omp flush(chunk_ready)
                #pragma omp flush(temp_dilated)
                
                for (int y = start_y; y < end_y; ++y) {
                    compute_erosion_for_row(temp_dilated, output, y, kernel_size);
                }
            }
        }
    }
    return output;
}




// Funzione che ottimizza l'erosione parallela separando il kernel 2D in due passaggi 1D (orizzontale e verticale)
// Configurazione Ottima: sfrutta il ciclo Row-Major, Collapse(2) e lo Scheduling Dinamico senza pragma SIMD manuali.
GrayImage erode_separable_parallel_optimal(const GrayImage& input, int kernel_size) {
    int w = input.getWidth();
    int h = input.getHeight();
    int offset = kernel_size / 2;
    
    GrayImage temp(w, h);
    GrayImage output(w, h);

    // Passaggio 1: Orizzontale
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            unsigned char min_val = 255;

            for (int kx = -offset; kx <= offset; ++kx) {
                int nx = std::max(0, std::min(w - 1, x + kx));
                min_val = std::min(min_val, input.getPixel(nx, y));
            }
            temp.setPixel(x, y, min_val);
        }
    }

    // Passaggio 2: Verticale
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            unsigned char min_val = 255;
            
            for (int ky = -offset; ky <= offset; ++ky) {
                int ny = std::max(0, std::min(h - 1, y + ky));
                min_val = std::min(min_val, temp.getPixel(x, ny));
            }
            output.setPixel(x, y, min_val);
        }
    }

    return output;
}

// Funzione che ottimizza la dilatazione parallela separando il kernel 2D in due passaggi 1D (orizzontale e verticale)
// Configurazione Ottima: sfrutta il ciclo Row-Major, Collapse(2) e lo Scheduling Dinamico senza pragma SIMD manuali.
GrayImage dilate_separable_parallel_optimal(const GrayImage& input, int kernel_size) {
    int w = input.getWidth();
    int h = input.getHeight();
    int offset = kernel_size / 2;
    
    GrayImage temp(w, h);
    GrayImage output(w, h);

    // Passaggio 1: Orizzontale
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            unsigned char max_val = 0;
            
            for (int kx = -offset; kx <= offset; ++kx) {
                int nx = std::max(0, std::min(w - 1, x + kx));
                max_val = std::max(max_val, input.getPixel(nx, y));
            }
            temp.setPixel(x, y, max_val);
        }
    }

    // Passaggio 2: Verticale
    #pragma omp parallel for collapse(2) schedule(dynamic)
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

GrayImage opening_separable_parallel_optimal(const GrayImage& input, int kernel_size) {
    GrayImage temp = erode_separable_parallel_optimal(input, kernel_size);
    return dilate_separable_parallel_optimal(temp, kernel_size);
}

GrayImage closing_separable_parallel_optimal(const GrayImage& input, int kernel_size) {
    GrayImage temp = dilate_separable_parallel_optimal(input, kernel_size);
    return erode_separable_parallel_optimal(temp, kernel_size);
}