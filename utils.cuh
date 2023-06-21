#ifndef SMILECOMPR_DICT_ITEM
#define SMILECOMPR_DICT_ITEM
#define MAX_PATT_LEN 15     //exclusive
#define MAX_BIN_LEN 32

__device__ __host__ void pushBack(char* string, char c);
__device__ __host__ void cudaPushBack(char* string, char c);

__device__ __host__ int myStrlen(char* string);
__device__ __host__ void myStrcpy(char* dest, const char* source);

typedef struct {
    unsigned char patt[MAX_PATT_LEN];
    unsigned int score;
    char bin[MAX_BIN_LEN];
    unsigned short bin_len;
}dict_item;
#endif


#ifndef SMILESCOMPR_TEXT_NODE
#define SMILESCOMPR_TEXT_NODE

typedef struct {
    unsigned char original;
    unsigned short encoded;
    short next;
}text_node;
#endif


#ifndef SMILESCOMPR_PATT_COUNTER
#define SMILESCOMPR_PATT_COUNTER

typedef struct {
    unsigned char patt[MAX_PATT_LEN];
    unsigned int count;
} patt_counter;
#endif

#ifndef SMILESCOMPR_PATT
#define SMILESCOMPR_PATT

typedef struct{
    unsigned char patt[MAX_PATT_LEN];
} pattern;
#endif

#ifndef SMILESCOMPR_UTILS_H
#define SMILESCOMPR_UTILS_H

#define MAX_SENTENCE_LEN 5000
#define PRINTABLE 128
#define ESCAPE ' '


#endif //SMILESCOMPR_UTILS_H
