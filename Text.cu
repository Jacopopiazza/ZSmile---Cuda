#include "Text.cuh"
#include <fstream>
#include "Preprocess.cuh"

using namespace std;

//external functions
short getLength(unsigned char* s);

bool string_compare(text_node * a, unsigned int len, unsigned char * b) {
    int j=0;
    while(j<len){
        if(a->original!= b[j]) return false;
        a++;
        j++;
    }
    return true;
}

/*
 *  --- BASIC FUNCTIONS --
 */
//constructors
Text::Text(char *input_file) {
    original_size=0;
    ifstream in(input_file);
    if(in.is_open()){
        string input_string;
        while(!in.eof()){
            getline(in, input_string);
            original_size += (input_string.size() +1);
            if(!input_string.empty()){
                auto * curr = (text_node *)malloc(input_string.size()*sizeof(text_node));
                text.push_back(curr);
                sentence_length.push_back(input_string.size());
                int i=0;
                while(i<input_string.size()){
                    curr->original=input_string[i];
                    curr->encoded=(unsigned char)input_string[i];
                    curr->next=1;
                    curr++;
                    i++;
                }
                //(curr-1)->next=0;
            }
        }
        in.close();
    }
    encoded_size=original_size;
}

Text::Text(bool preprocess) {
    Preprocess p(false);
    original_size=0;

    string input;
    char input_string[MAX_SENTENCE_LEN];
    int size;
    while(getline(std::cin, input)){
        if(preprocess) {
            p.preprocess_ring(input, input_string);
            size = strlen(input_string);
        }else{
            for(int i=0; i<input.size(); i++){
                input_string[i]=input[i];
            }
            input_string[input.size()]='\0';
            size = input.size();
        }
        original_size += (size +1);
        if(size>0){
            auto * curr = (text_node *)malloc(size*sizeof(text_node));
            text.push_back(curr);
            sentence_length.push_back(size);
            int i=0;
            while(i<size){
                curr->original=input_string[i];
                curr->encoded=(unsigned char)input_string[i];
                curr->next=1;
                curr++;
                i++;
            }
            //(curr-1)->next=0;
        }
    }
    encoded_size=original_size;
}

//destructors
Text::~Text() {
    for(auto & i : text){
        free(i);
    }
    //text.clear();
}

void Text::print(char *output_file) {
    ofstream enc(output_file);
    if(enc.is_open()) {
        for (int i = 0; i < text.size(); i++) {
            text_node *curr = text[i];
            text_node * end = curr + sentence_length[i];
            while (curr < end) {
                enc << curr->encoded;
                curr += curr->next;
            }
            //enc << curr->encoded;
            if (i < text.size() - 1) enc << endl;
        }
    }
}

//print to std::cout
void Text::print() {
    for (int i = 0; i < text.size(); i++) {
        text_node *curr = text[i];
        text_node * end = curr + sentence_length[i];
        while (curr < end) {
            std::cout << curr->encoded;
            curr += curr->next;
        }
        //enc << curr->encoded;
        if (i < text.size() - 1) std::cout << endl;
    }
}

/*
 * --- HASH TABLE MODIFIER
 */
void Text::find_all_patterns(int min_size, int max_size, HashTable *hashTable) {
    auto* last_patterns = (unsigned char*)calloc(max_size*max_size, sizeof(unsigned char));
    int insert =0;
    for(int length = min_size; length < max_size; length++){
        for(int i=0; i< text.size(); i++){
            text_node * curr=text[i];
            //get size of the string
            int size=sentence_length[i];

            for(int j=0; j < size - length +1; j++){
                //verify overlapping
                bool overlapping = false;
                for(int k=0; k< length -1 && ! overlapping; k++)
                    if(string_compare(curr, length, last_patterns + k*max_size))
                        overlapping=true;
                if(!overlapping) hashTable->add_new_item(curr, length); //add to hash table

                //update last_patterns
                text_node * temp=curr;
                for(int k=0; k<length; k++) {
                    last_patterns[insert * max_size + k] = temp->encoded;
                    temp += temp->next;
                }
                insert++;
                insert%=(length-1);

                curr += curr->next;
            }
            //clear last_patterns
            for(int t=0; t < max_size*max_size; t++) last_patterns[t]='\0';
        }
    }

}

/*
 *  --- MODIFIER ---
 */
void Text::update(patt_counter patt, unsigned int code, HashTable* hashTable, Dictionary* dictionary){
    short length = getLength(patt.patt);

    for(int i=0; i< text.size(); i++) {
        //parellalizable for each sentence
        text_node *curr = text[i];
        text_node *sentence_start = text[i];
        text_node * last_valid = sentence_start + sentence_length[i] -1;
        int pos = find(curr, patt, last_valid);
        while (pos >= 0) {
            //find an occurency in original test, need to check if it is convinient to
            curr += pos;
            int saved_char = check(curr, length);
            if (saved_char > 0) {
                encoded_size -= saved_char;
                dictionary->increase(code, length);
                fillPP(sentence_start, curr, last_valid, length);
                //update text and dictionary
                int real_saved = updateTextDictionary(curr, dictionary, code, length);
                if(saved_char != real_saved) {
                    int breakpoint2 = 0;
                }
                updateTable(hashTable);
            }
            curr++;
            pos = find(curr, patt, last_valid);
        }
    }
}

void Text::remove(unsigned int code, short length, HashTable* hashTable) {
    for (int i = 0; i < text.size(); i++) {
        text_node *curr = text[i];
        text_node *sentence_start = curr;
        text_node * end = curr + sentence_length[i];
        while (curr < end) {
            if(curr->encoded==code){
                fillPP(sentence_start, curr, end -1, length);
                //text_node * previous_start = curr;
                encoded_size += (length -1);
                curr->encoded=curr->original;
                curr->next=1;
                curr++;
                while(curr->next <0){
                    curr->next=1;
                    curr++;
                }
                //correct(previous_start, curr);
                updateTable(hashTable);
            }
            else {
                curr += curr->next;
            }
        }
    }
}

/*
 *  --- GETTER ---
 */


unsigned long Text::getOriginalSize() const {
    return original_size;
}

unsigned long Text::getEncodedSize() const {
    return encoded_size;
}

unsigned long Text::getSavedSize() const{
    return original_size - encoded_size;
}

float Text::getRatio() const {
    return (float) encoded_size / (float) original_size;
}

unsigned int* Text::calculate_frequencies() {
    auto* occ = (unsigned int*)malloc(DICT_SIZE * sizeof(unsigned int));
    for(int i=0; i<DICT_SIZE; i++){
        occ[i]=0;
    }
    for(int i=0; i<text.size(); i++){
        text_node * curr=text[i];
        text_node * last_valid = curr + sentence_length[i];
        while(curr < last_valid){
            occ[curr->encoded]++;
            curr += curr->next;
        }
    }
    occ[ESCAPE]=text.size()/100;    //just an estimated value
    return occ;
}

/*
 *  --- PRIVATE FUNCTION ---
 */

/*
 * --- GENERAL POURPUSE ---
 */

int Text::find(text_node *start, patt_counter patt, text_node * last_valid) {
    //naive algorithm
    //TODO implements Knuth-Morris-Pratt
    int res=0;
    bool match=false;
    while(start<last_valid && !match){
        int i=0;
        while((start+i) <= last_valid && (start+i)->original==patt.patt[i]){
            i++;
        }
        if(patt.patt[i]=='\0'){
            match=true;
        }
        res++;
        start++;
    }
    if(match) return (res -1);
    else return -1;
}

int Text::check(text_node *start, int length) {
    //return the number of saved character by substituting the pattern locally
    //if res>0 substitution should be performed
    //in case of overlapping with initial pattern next will bring back current!
    int res= length -1;     //maximum saved_character value
    text_node* curr=start;
    while(curr - start < length){
        if(curr->next>1){
            //pattern deleting
            res -= (curr->next -1);
        }
        curr+=curr->next;
    }

    return res;
}

/*
 * TEXT update helper
 */
int Text::updateTextDictionary(text_node *pattern, Dictionary *dictionary, unsigned int code,  short  length){
    int real_saved_values=1;
    text_node * curr = pattern;
    if(curr->next<0){
        // initial overlapping with overlapping with previous pattern
        curr += curr->next;     //go to the starting point of the previous pattern
        text_node * previous_start = curr;
        curr->encoded = dictionary->reversePatt(curr->encoded);    //update encoded text
        real_saved_values -= (curr->next -1);
        curr->next=1;               //update next values
        curr++;
        while(curr < pattern){
            curr->next=1;
            curr++;
        }
        //correct(previous_start, curr);          //check for possible best pattern for the update text
    }
    //pattern start
    if(curr->next> 1){
        dictionary->reversePatt(curr->encoded);
        real_saved_values -= (curr->next-1);
    }
    curr->encoded = code;
    curr->next = length;

    curr++;

    for(short i=1; i<(length-1); i++){              //first and last character are managed by the two if
        if(curr->next > 1){
            curr->encoded = dictionary->reversePatt(curr->encoded); //restore original value for encoded
            real_saved_values -= (curr->next -1);
        }
        curr->next= (short)-i;
        curr++;
        real_saved_values++;
    }

    if(curr->next != 1){
        //final overlapping with previous pattern
        text_node * end_point= curr;
        if(curr->next >1){
            curr->encoded = dictionary->reversePatt(curr->encoded);
            real_saved_values -= (curr->next -1);
            curr++;
        }
        while(curr->next<0){
            curr->next=1;
            curr++;
        }
        //correct(end_point +1, curr);       //check for possible best pattern for the update text
        curr = end_point;
    }

    curr->next=(short)-(length -1);

    return real_saved_values;
}


/*
 *  --- HASHTABLE update helpers
 */

void Text::fillPP(text_node *sentence_start, text_node *first_changing, text_node *sentence_end, short length) {
    //nice function OCISLY
    text_node * last_changing = first_changing + length -1;        //last character of the original pattern

    //check previous pattern overlapping
    if(first_changing -> next < 0){
        first_changing += first_changing ->next;
    }

    //check final pattern overlapping
    if(last_changing -> next <0){
        while (last_changing <= (sentence_end+1) && last_changing->next <0) last_changing++;
        last_changing--;
    }
    else if(last_changing->next>1){
        last_changing += (last_changing->next -1);
    }

    text_node *curr = max(first_changing - MAX_PATT_LEN +2, sentence_start);
    //text_node * curr = sentence_start;
    //last_changing = sentence_end;
    previous_patterns_size=0;
    while(curr <= last_changing) {
        short i0 = max((long)MIN_PATT_LEN, first_changing - curr + 1);
        short i1 = min((long) MAX_PATT_LEN, sentence_end - curr + 2);
        for (short i = i0; i < i1; i++) {
        //for(int i= MIN_PATT_LEN; i<min((long) MAX_PATT_LEN, sentence_end - curr +2); i++){
            if(!overlapping(curr, i, sentence_start)){
                previous_patterns[previous_patterns_size].score = check(curr, i);;
                previous_patterns[previous_patterns_size].len=i;
                previous_patterns[previous_patterns_size].pos=curr;
                previous_patterns_size++;
            }
        }
        curr++;
    }

}

void Text::updateTable(HashTable * hashTable){
    for(int i=0; i<previous_patterns_size; i++){
        int new_score = check(previous_patterns[i].pos, previous_patterns[i].len);
        if(new_score>0){
            if(new_score>previous_patterns[i].score){
                hashTable -> add(previous_patterns[i].pos, previous_patterns[i].len, new_score - max(0, previous_patterns[i].score));
            }
            else {
                hashTable->sub(previous_patterns[i].pos, previous_patterns[i].len, previous_patterns[i].score - new_score);
            }
        }
        else{
            if(previous_patterns[i].score>0)
                hashTable->sub(previous_patterns[i].pos, previous_patterns[i].len, previous_patterns[i].score);
        }
    }

}


bool Text::overlapping(text_node *pattern, short length, text_node *sentence_start) {
    text_node * curr= max(sentence_start, pattern - length +1);
    while(curr<pattern){
        int i=0;
        while(i<length && (curr+i)->original == (pattern+i)->original) i++;
        if(i==length)
            return true;
        curr++;
    }

    return false;
}