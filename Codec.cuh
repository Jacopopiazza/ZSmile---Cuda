#ifndef SMILESCOMPRESSION_CODEC_H
#define SMILESCOMPRESSION_CODEC_H
#include <string.h>
#include <string>
#include "Dictionary.cuh"
#include "Preprocess.cuh"

typedef struct node{
    node* next[PRINTABLE];
    bool isValid;
    unsigned int code;
}node;

typedef struct decNode{
    unsigned int c;
    decNode* left;  //0
    decNode* right; //1
}decNode;

class Codec {
public:
    Codec(const std::string& dictionary_filename, bool preprocess, bool permessive, bool huffman);
    static Codec* cudaCodec(const std::string& dictionary_filename, bool preprocess, bool permissive, bool huffman);
    ~Codec();
    __device__ __host__ int smile_compress(char* smile_in, char* smile_zip,int *scoreMatrix, unsigned int *bestMatrix,unsigned int *all_matchMatrix, char *sMatrix);

    __device__ __host__ int smile_decompress(char* smile_zip, char* smile_out);

    int getMaxLengthOfDictionaryPattern();

private:
    __device__ __host__ int smile_decompress_std(char* smile_zip, char* smile_out);
    int smile_decompress_huffman(char* smile_zip, char* smile_out);

    __device__ __host__ node* build_tree();
    decNode* build_dec_tree();
    void magic_generate();

    __device__ __host__ void delete_tree(node* n);
    void delete_dec_tree(decNode *root);

    //option
    bool preprocess;
    bool huffman;


    //data generated
    Dictionary* dictionary;
    Preprocess* p;
    int * length;
    int* weight;
    node* compression_tree;
    char* table;
    decNode* huffman_tree;
    std::string huffman_code[DICT_SIZE];

    /*int score[MAX_SENTENCE_LEN]{};
    unsigned int best[MAX_SENTENCE_LEN]{};
    unsigned int all_match[MAX_SENTENCE_LEN*MAX_PATT_LEN]{};
    char s[MAX_SENTENCE_LEN]{};*/

    unsigned char magic_sequence[8]{};


};


#endif //SMILESCOMPRESSION_CODEC_H
