#include "Preprocess.cuh"
#include "utils.cuh"
#include <stdio.h>

using namespace std;

Preprocess::Preprocess(bool permissive_ext) {
    //for(int i=1; i<MAX_RING; i++) free_value[i]=true;
    //free_value[0]=false;
    //for(int i=1; i<MAX_RING; i++) opened[i] = {-1, false, 0};
    //opened[0]= {-1, true, 0};

    this->permissive = permissive_ext;
}

Preprocess* Preprocess::cudaPreprocess(bool permissive_ext) {
    Preprocess* p;
    cudaMallocManaged(&p, sizeof(Preprocess));
    //for(int i=1; i<MAX_RING; i++) p->free_value[i]=true;
    //p->free_value[0]=false;
    //for(int i=1; i<MAX_RING; i++) p->opened[i] = {-1, false, 0};
    //opened[0]= {-1, true, 0};

    p->permissive = permissive_ext;

    return p;
}

int Preprocess::preprocess_ring(char* s, char* s_out){
    bool free_value[MAX_RING];
    number opened[MAX_RING];

    for(int i=1; i<MAX_RING; i++) free_value[i]=true;
    free_value[0]=false;
    for(int i=1; i<MAX_RING; i++) opened[i] = {-1, false, 0};


    int first_free = 1;
    int last_open = -1;
    int j = 0;        //index for s_out
    bool skip = false;

    int len;
    len = myStrlen(s);

    for (int i = 0; i < len; i++) {
        if(s[i]=='[')
            skip=true;
        if(s[i]==']')
            skip=false;

        if (((s[i] >= '0' && s[i] <= '9') || (s[i]=='%'))&& !skip) {
            int curr;
            if(s[i]=='%'){
                curr = s[i+1] *10;
                curr += s[i+2]%10;
                i+=2;
            }else{
                curr = s[i] - '0';
            }

            if (curr < MAX_RING && !opened[curr].open) {
                last_open = curr;
                opened[curr] = {j, true, first_free};
                print(s_out, &j, first_free);
                free_value[first_free]= false;
                while(!free_value[first_free]) first_free++;
            } else {
                if (curr == last_open) {
                    print(s_out, &j, 0);
                    correct(s_out, &j, opened[curr].index);
                } else {
                    print(s_out, &j, opened[curr].encoded);
                }
                last_open = -1;
                free_value[opened[curr].encoded]=true;
                if(opened[curr].encoded<first_free)
                    first_free=opened[curr].encoded;
                opened[curr]={-1, false, 0};

            }
        }else {
            s_out[j] = s[i];
            j++;
        }
    }
    //for (; j < MAX_STR_LEN; j++)
    s_out[j] = '\0';
    if(first_free!=1 || last_open!=-1){
        //invalid smiles
        //reset
        for(int i=1; i<MAX_RING; i++) free_value[i]=true;
        free_value[0]=false;
        for(int i=1; i<MAX_RING; i++) opened[i] = {-1, false, 0};
        return -1;
    }
    return 0;
}

int Preprocess::postprocess_ring(const char* s, int size, char* result){
    bool free_value[MAX_RING];
    number opened[MAX_RING];
    char buffer_out[MAX_SENTENCE_LEN];

    for(int i=1; i<MAX_RING; i++) free_value[i]=true;
    free_value[0]=false;
    for(int i=1; i<MAX_RING; i++) opened[i] = {-1, false, 0};


    int j = 0;        //index for s_out
    bool skip = false;
    int next_number = 1;
    bool last_changing = false;     //for strange CheMBL policy

    for (int i = 0; i < size; i++) {
        if(s[i]=='[')
            skip=true;
        if(s[i]==']')
            skip=false;

        if (((s[i] >= '0' && s[i] <= '9') || (s[i]=='%'))&& !skip) {
            int curr;
            if(s[i]=='%'){
                curr = s[i+1] *10;
                curr += s[i+2]%10;
                i+=2;
            }else{
                curr = s[i] - '0';
            }


            if (curr < MAX_RING && !opened[curr].open) {
                print(buffer_out, &j, next_number);
                opened[curr] = {j, true, next_number};
                free_value[next_number]=false;
                while(!free_value[next_number]) next_number++;
            } else {
                print(buffer_out, &j, opened[curr].encoded);
                free_value[opened[curr].encoded]=true;
                opened[curr] = {-1, false, 0};
                last_changing=true;
            }

        }else {
            buffer_out[j] = s[i];
            j++;
            if(last_changing && permissive){
                for(int i=next_number-1; i>0; i--){
                    if(free_value[i])
                        next_number=i;
                }
            }
            last_changing=false;
        }
    }

    buffer_out[j] = '\0';
    result[0] = '\0';
    
    myStrcpy(result, buffer_out);

    return 0;
}


/*
 *
 *      --- PRIVATE FUNCTIONS --
 *
 */
void Preprocess::print(char* out, int * j, int val){
    if(val < 10){
        out[*j]= '0' + val;
        (*j)++;
    }
    else{
        out[*j] = '%';
        out[*j + 1] = '0' + val / 10;
        out[*j + 2] = '0' + val % 10;
        *j+=3;
    }
}
void Preprocess::correct(char* out, int * end, int pos){
    if(out[pos] == '%'){
        for(int i=pos+3; i<*end; i++){
            out[i - 2]=out[i];
        }
        *end-=2;
    }
    out[pos]='0';
}
