#ifndef SMILESCOMPR_HASHTABLE_H
#define SMILESCOMPR_HASHTABLE_H

#include <vector>
#include "utils.cuh"


#define TABLE_SIZE 200000000       //hash table size 4GB


class HashTable {
public:
    HashTable();
    ~HashTable();

    //modifer
    void add_new_item(text_node * a, unsigned int len);
    void sub(text_node *a, unsigned int len, int offset);
    void add(text_node *a, unsigned int len, int offset);
    void clear();

    //getter
    bool safe_delete();
    patt_counter getMax();
    patt_counter* getMax(int n);
    unsigned int getUsage();


private:
    patt_counter *table;
    unsigned int usage;
    
    //temporary variable
    int old_max;
    
    unsigned int hash_str(text_node * a, unsigned int len);
    static bool contains(const std::vector<int>& used_index, int index);
};


#endif //SMILESCOMPR_HASHTABLE_H
