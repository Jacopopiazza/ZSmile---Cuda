#ifndef SMILESCOMPR_DICTIONARY_H
#define SMILESCOMPR_DICTIONARY_H
#define FIRST_CHAR '!'     //inclusive
#define MIN_PATT_LEN 2      //inclusive

#define DICT_SIZE 1024

#include <string>
#include <iostream>
#include "utils.cuh"

//OOP version


typedef struct Hnode{
    unsigned int c;
    unsigned int count;
    Hnode* left_child;     //0
    Hnode* right_child;    //1
}Hnode;

class Dictionary {
public:
    //constructor, destructor and load functions
    explicit Dictionary();
    static Dictionary* cudaDictionary();
    ~Dictionary();
    void load_alphabet(const char* alphabet);
    void load_alphabet();
    void load_dictionary(const char* dictionary_file);
    bool set_size(unsigned int size);
    void print_to_file(char *dictionary_file);
    void print_to_std_out();
    char *print_table();
    int *get_length();
    int get_max_length();
    int * get_bin_length();

    //modifier
    void add(patt_counter c);
    void increase(unsigned int c, short length);
    unsigned char reversePatt(unsigned int c);
    patt_counter removeCode(unsigned int code);


    //getter
    unsigned int getCode() const;
    bool isFull() const;
    unsigned int getLower();
    unsigned int getScore(unsigned int c);
    unsigned long getTotalScore();
    unsigned int getLowerCode();
    int getBinLen(unsigned int code_req);
    pattern getPatt(unsigned int code_Req);
    short getPattLen(unsigned int code_Req);
    std::string getBinCode(unsigned int code_Req);
    int getSize();


    //HUffman
    void compute_Huffman(unsigned int * occ);

private:
    dict_item * dictionary;
    bool used_char[DICT_SIZE]{};
    unsigned int code;
    unsigned int size{};

    void update_code();
    bool isValid(unsigned int a);
    void clear_dictionary();

    //Huffman Functions
    static bool loe(Hnode* a, Hnode* b);
    void build_Huffman_tree(const unsigned int* occ);
    void write_Huffman(Hnode* n, std::string s);



};


#endif //SMILESCOMPR_DICTIONARY_H
