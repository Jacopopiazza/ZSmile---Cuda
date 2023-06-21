#include "HashTable.cuh"
#include <cstdlib>
#include <iostream>

//external functions
bool string_compare(text_node * a, unsigned int len,unsigned char * b);

bool subpattern(unsigned char * small, unsigned char * big){
    bool res = false;
    while(*big!='\0' && !res){
        int i=0;
        while(big[i]!='\0' && big[i]==small[i]) i++;

        if(small[i]=='\0')
            res=true;

        big++;
    }
    return res;
}
/*
 *  --- BASIC FUNCTIONS ---
 */
HashTable::HashTable() {
    table=(patt_counter*) calloc(TABLE_SIZE, sizeof(patt_counter));
    usage=0;
    old_max=0;
    clear();
}

HashTable::~HashTable() {
    free(table);
}

/*
 * --- MODIFIER ---
 */
void HashTable::add_new_item(text_node * a, unsigned int len){
    unsigned int index=hash_str(a, len)%TABLE_SIZE;

    if(usage>TABLE_SIZE*0.9){
        std::cout<<"No more space"<<std::endl;
        exit(0);
    }
    if(table[index].patt[0]=='\0'){
        int j=0;
        while(j<len) {
            table[index].patt[j]=a->encoded;
            a += a->next;
            j++;
        }
        table[index].patt[j]='\0';
        usage++;
    }
    table[index].count+=(len -1);
}

void HashTable::sub(text_node *a, unsigned int len, int offset) {
    unsigned int index=hash_str(a, len)%TABLE_SIZE;
    if(offset>table[index].count)
        int breakpoitn =0;
    table[index].count -= offset;
}

void HashTable::add(text_node *a, unsigned int len, int offset) {
    unsigned int index=hash_str(a, len)%TABLE_SIZE;
    table[index].count += offset;
}

void HashTable::clear() {
    for(int i=0; i< TABLE_SIZE; i++){
        table[i].count=0;
        table[i].patt[0] = '\0';
    }
}
/*
 *  --- GETTER ---
 */
bool HashTable::safe_delete() {
    bool res = table[old_max].count<=usage;
    if(!res) table[old_max].count=0;
    return res;
}

patt_counter HashTable::getMax() {
    patt_counter max;
    max.count=0;
    for(int i=0; i< TABLE_SIZE; i++){
        if(table[i].count<0){
            int breakpoint=0;
        }
        if(table[i].count>max.count){
            old_max=i;
            max.count=table[i].count;
            for(int j=0; j< MAX_PATT_LEN; j++)
                max.patt[j] = table[i].patt[j];
        }
        //table[i].count=0;
        //table[i].patt[0] = '\0';
    }
    return max;
}

patt_counter* HashTable::getMax(int n) {
    auto* res = (patt_counter*)malloc(sizeof(patt_counter)*n);
    std::vector<int> used_index;
    int j=0;
    while(j<n) {
        int index;
        unsigned int max = 0;
        for (int i = 0; i < TABLE_SIZE; i++) {
            if (table[i].count > max) {
                if (!contains(used_index, i)) {
                    max = table[i].count;
                    index = i;
                }
            }
        }
        used_index.push_back(index);
        bool fail = false;
        int i=0;
        while(!fail && i<j){
            if(subpattern(table[index].patt, (res +  i)->patt))
                fail=true;
            i++;
        }
        if(!fail){
            res[j]=table[index];
            j++;
        }
    }
    return res;
}

unsigned int HashTable::getUsage(){
    return usage;
}

/*
 *  --- PRIVATE ---
 */
unsigned int HashTable::hash_str(text_node * a, unsigned int len) {
    text_node * start= a;
    unsigned int h = 37;
    int j=0;
    while (j<len) {
        h = (h * 54059) ^ (a->original * 76963);
        j++;
        a ++;
        h%=TABLE_SIZE;
    }
    while(table[h].patt[0]!='\0' && ! string_compare(start, len, table[h].patt)){
        h++;
        h%=TABLE_SIZE;
        //cout<<"hash_fail: "<< index;
    }
    return h;
}

bool HashTable::contains(const std::vector<int>& used_index,int  index){
    for(int x : used_index){
        if(x==index) return true;
    }
    return false;
}


