#ifndef SMILESCOMPRESSION_PREPROCESS_H
#define SMILESCOMPRESSION_PREPROCESS_H
#include <string.h>
#include "utils.cuh"
#define MAX_RING 500

class Preprocess {
public:
    explicit Preprocess(bool permissive);
    static Preprocess* cudaPreprocess(bool permissive_ext);
    __device__ __host__ int preprocess_ring(char* s, char* s_out);
    __device__ __host__ int postprocess_ring(const char* s, int size, char* result);

private:
    bool permissive;
    typedef struct{
        int index;
        bool open;
        int encoded;
    } number;
    

    __device__ __host__ void print(char* out, int * j, int val);
    __device__ __host__  void correct(char* out, int * end, int pos);
};


#endif //SMILESCOMPRESSION_PREPROCESS_H
