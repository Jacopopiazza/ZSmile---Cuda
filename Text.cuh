#ifndef SMILESCOMPR_TEXT_H
#define SMILESCOMPR_TEXT_H

#include "utils.cuh"
#include <vector>
#include <iostream>
#include <cstring>
#include "HashTable.cuh"
#include "Dictionary.cuh"

class Text{

public:
    //basic functions
    explicit Text(char * input_file);
    explicit Text(bool preprocess);
    ~Text();
    void print(char * output_file);
    void print();

    //hashTable modifier
    void find_all_patterns(int min_size, int max_size, HashTable *  hashTable);

    //modifier
    void update(patt_counter patt, unsigned int code, HashTable* hashTable, Dictionary* dictionary);
    void remove(unsigned int code, short length, HashTable* hashTable);

    //getter
    unsigned long getOriginalSize() const;
    unsigned long getEncodedSize() const;
    unsigned long getSavedSize() const;
    float getRatio() const;
    unsigned int * calculate_frequencies();

private:
    std::vector<text_node*> text;
    std::vector<int> sentence_length;
    unsigned long int original_size;
    unsigned long int encoded_size;

    //temporary variable
    typedef struct{
        text_node * pos;
        int len;
        int score;
    }pp;    //previous pattern
    pp previous_patterns[(3*MAX_PATT_LEN -5)*(MAX_PATT_LEN -1)];
    int previous_patterns_size;     //first free element of previous patterns


    static int find(text_node* start, patt_counter patt, text_node* last_valid);
    static int check(text_node * start, int length);

    //text update helper functions
    static int updateTextDictionary(text_node *pattern, Dictionary *dictionary,unsigned int code, short length);

    //hashTable update helper functions
    void fillPP(text_node *sentence_start, text_node *first_changing, text_node *sentence_end, short length);
    void updateTable(HashTable *hashTable);
    static bool overlapping(text_node * pattern, short length, text_node * sentence_start);

};


#endif //SMILESCOMPR_TEXT_H
