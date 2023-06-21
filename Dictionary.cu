#include "Dictionary.cuh"
#include <fstream>
#include <algorithm>
#include <vector>
#include <cstring>
//OOP version
using namespace std;

//external functions
short getLength(unsigned char* s);

/*
 *  --- BASIC FUNCTIONS --
 */
//constructor
Dictionary::Dictionary() {
    code = FIRST_CHAR;
    //initialize used_char array
    for(bool & i : used_char) i=false;



    //initialize dictionary
    dictionary=(dict_item *)calloc(DICT_SIZE, sizeof(dict_item));
    for(int i=0; i<DICT_SIZE; i++){
        dictionary[i].patt[0]='\0';
        dictionary[i].score=0;
    }
}

Dictionary* Dictionary::cudaDictionary(){
    Dictionary* d;
    cudaMallocManaged(&d, sizeof(Dictionary));

    d->code = FIRST_CHAR;
    //initialize used_char array
    for(bool & i : d->used_char) i=false;



    //initialize dictionary
    cudaMallocManaged(&(d->dictionary), DICT_SIZE*sizeof(dict_item));

    for(int i=0; i<DICT_SIZE; i++){
        d->dictionary[i].patt[0]='\0';
        d->dictionary[i].score=0;
    }

    return d;
}

//destructor
Dictionary::~Dictionary() {

    cudaFree(dictionary);
    
}

void Dictionary::load_alphabet(const char *alphabet) {
    used_char[ESCAPE]=true;
    //compute used_char array
    ifstream alp(alphabet);
    if(alp.is_open()) {
        int n;
        char c;
        while (!alp.eof()) {
            alp >> n;
            used_char[n] = true;
            if (n > 32) alp >> c;
            //if(c!=n) exit(0);
        }
        alp.close();
    }

    for(short i=0; i<DICT_SIZE; i++){
        if(isValid(i)){
            dictionary[i].patt[0]=(char)i;
            dictionary[i].patt[1]='\0';
        }
    }

    while(isValid(code)) code++;
}

void Dictionary::load_alphabet() {
    //compute used_char array default dictionary
    for(unsigned char i = '!'; i < PRINTABLE; i++)
        used_char[i]=true;

    for(short i=0; i<DICT_SIZE; i++){
        if(isValid(i)){
            dictionary[i].patt[0]=(char)i;
            dictionary[i].patt[1]='\0';
        }
    }

    while(isValid(code)) code++;
}

void Dictionary::load_dictionary(const char *dictionary_file) {
    ifstream in(dictionary_file);
    int input_code;
    char c;
    while(!in.eof()){
        in>>input_code;
        in.get(c); //expected newline
        if(input_code =='\n'){
            in.get(c); //expected another newline
            dictionary[input_code].patt[0]='\n';
        }
        int i=0;
        in.get(c);
        while(c!='\n'){
            dictionary[input_code].patt[i]=c;
            i++;
            in.get(c);
        }
        in>>dictionary[input_code].bin;
        int l=0;
        while(dictionary[input_code].bin[l]!='\0') l++;
        dictionary[input_code].bin_len=l;
    }

    //for(int i=0; i<dictionary.size(); i++){
    //    cout<<dictionary[i].patt<< " ";
    //    cout<<(int)dictionary[i].code<<endl;
    //}

    //clear_dictionary();
}

bool Dictionary::set_size(unsigned int external_size){
    if(external_size>DICT_SIZE) return false;
    size=external_size;
    return true;
}

void Dictionary::print_to_file(char * dictionary_file) {
    ofstream out(dictionary_file);
    if(out.is_open()) {
        for (int i = 0; i < DICT_SIZE; i++) {
            if (dictionary[i].patt[0] != '\0') {
                out << i << endl;
                out << dictionary[i].patt << endl;
                out << dictionary[i].bin << endl;
                if (i < DICT_SIZE - 1) out << endl;
            }
        }
        out.close();
    }
}

void Dictionary::print_to_std_out() {
    for (int i = 0; i < DICT_SIZE; i++) {
        if (dictionary[i].patt[0] != '\0') {
            std::cout << i << endl;
            std::cout << dictionary[i].patt << endl;
            std::cout << dictionary[i].bin << endl;
            if (i < DICT_SIZE - 1) std::cout << endl;
        }
    }
}

char* Dictionary::print_table(){
    //char* res = (char*)malloc(DICT_SIZE*MAX_PATT_LEN);
    char *res;
    cudaMallocManaged(&res, DICT_SIZE*MAX_PATT_LEN*sizeof(char));

    for(int i=0; i<DICT_SIZE; i++){
        int j=0;
        while(j< MAX_PATT_LEN && dictionary[i].patt[j]!='\0') {
            res[i * MAX_PATT_LEN + j] = dictionary[i].patt[j];
            j++;
        }
        res[i * MAX_PATT_LEN + j]='\0';
    }
    return res;
}

int Dictionary::get_max_length(){
    int  res=-1;
    

    for(int i=0; i<DICT_SIZE; i++){
        if(getLength(dictionary[i].patt) > res){
            res = getLength(dictionary[i].patt);
        }
    }
    return res;
}

int* Dictionary::get_length(){
    int * res;// = (int*)malloc(DICT_SIZE*sizeof(int));
    cudaMallocManaged(&res, DICT_SIZE*sizeof(int));

    for(int i=0; i<DICT_SIZE; i++){
        res[i]=getLength(dictionary[i].patt);
    }
    return res;
}

int *Dictionary::get_bin_length() {
    int * res;// = (int*)malloc(DICT_SIZE*sizeof(int));
    cudaMallocManaged(&res, DICT_SIZE*sizeof(int));


    for(int i=0; i<DICT_SIZE; i++){
        res[i]=strlen(dictionary[i].bin);
    }
    return res;
}

/*
 *      --- CHANGING FUNCTIONS ---
 */

//increasing score
//require !isFull()
void Dictionary::add(patt_counter c) {
    //add a new entry to dictionary
    short length = getLength(c.patt);
    for(int i=0; i< length; i++) {
        dictionary[code].patt[i] = c.patt[i];
    }
    dictionary[code].patt[length]='\0';
    //dictionary[code].score=c.count;
    dictionary[code].score=0;
    update_code();
}

void Dictionary::increase(unsigned int c, short length) {
    dictionary[c].score += (length -1);
}

//decreasing score
unsigned char Dictionary::reversePatt(unsigned int c) {
    int l= getLength(dictionary[c].patt);
    dictionary[c].score -= (l-1);
    return dictionary[c].patt[0];
}

patt_counter Dictionary::removeCode(unsigned int external_code){
    patt_counter res;
    res.count = dictionary[external_code].score;
    int i=0;
    while(dictionary[external_code].patt[i]!='\0'){
        res.patt[i]=dictionary[external_code].patt[i];
        i++;
    }
    res.patt[i]='\0';
    code=external_code;
    used_char[external_code]=false;
    dictionary[external_code].patt[0]='\0';
    dictionary[external_code].score=0;
    return res;
}


/*
 *  --- GETTER ---
 */

unsigned int Dictionary::getCode() const {
    //get the first code free to use in dictionary
    return code;
}

bool Dictionary::isFull() const {
    //return true <=> You can add a new entry to dictionary
    return code >= size;
}

unsigned int Dictionary::getLower() {
    unsigned int min=0;
    for(int i=0; i<DICT_SIZE; i++){
        if(min==0) min = dictionary[i].score;
        if(dictionary[i].score >0 && dictionary[i].score < min)
            min=dictionary[i].score;
    }
    return min;
}

unsigned int Dictionary::getScore(unsigned int c) {
    return dictionary[c].score;
}

unsigned long Dictionary::getTotalScore() {
    unsigned long res=0;
    for(int i=0; i<DICT_SIZE; i++) res+= dictionary[i].score;
    return res;
}

unsigned int Dictionary::getLowerCode() {
    unsigned int min=0;
    unsigned char res;
    for(int i=0; i<DICT_SIZE; i++) {
        if (min == 0) {
            min = dictionary[i].score;
            res = i;
        }
        if (dictionary[i].score > 0 && dictionary[i].score < min){
            min = dictionary[i].score;
            res = i;
        }
    }
    //unsigned char original = dictionary[res].patt[0];
    //short length = getLength(dictionary[res].patt);
    return res;
}

int Dictionary::getBinLen(unsigned int code_req) {
    return dictionary[code_req].bin_len;
}

pattern Dictionary::getPatt(unsigned int code_req) {
    auto l=(short)(getLength(dictionary[code_req].patt)+1);
    pattern res;
    for(int i=0; i<l; i++){
        res.patt[i]=dictionary[code_req].patt[i];
    }
    return res;
}

short Dictionary::getPattLen(unsigned int code_Req) {
    return getLength(dictionary[code_Req].patt);
}

int Dictionary::getSize(){
    for(int i=128; i<DICT_SIZE; i++){
        if(dictionary[i].patt[0]=='\0')
            return i;
    }
    return DICT_SIZE;
}

/*
 *  --- HUFFMAN ---
 */

void Dictionary::compute_Huffman(unsigned int *occ) {
    //check for missing value
    for(int i=0; i<256; i++){
        if(isValid(i) && occ[i]==0) occ[i]=1;
    }
    if(!isValid(ESCAPE)) occ[ESCAPE]=0;
    build_Huffman_tree(occ);
}


/*
 *     --- PRIVATE FUNCTIONS ---
 */

void Dictionary::update_code() {
    //Called after add
    used_char[code]=true;
    code++;
    while(isValid(code) && !isFull()) code++;
}


bool Dictionary::isValid(unsigned int a) {
    //return true <=> a is an original character(included in alphabet)
    //return a < 127
    return used_char[a];
}

void Dictionary::clear_dictionary() {
    //compute dictionary pattern just using original characters
    for(int i=0; i<DICT_SIZE; i++){
        if(dictionary[i].patt[0]!='\0' && dictionary[i].patt[0]!=i) {
            int l=0;
            while(dictionary[i].patt[l]!='\0') l++;
            for (int j = i + 1; j < DICT_SIZE; j++) {
                int k = 0;
                while(dictionary[j].patt[k]!='\0'){
                    if(dictionary[j].patt[k]==i){
                        int z=k;
                        while(dictionary[j].patt[z]!='\0') z++;
                        for(z--; z>k; z--)
                            dictionary[j].patt[z+l-1]=dictionary[j].patt[z];
                        for(int w=0; w<l; w++){
                            dictionary[j].patt[k+w]=dictionary[i].patt[w];
                        }
                        k+=(l-1);
                    }
                    k++;
                }
            }
        }
    }

    //for(int i=0; i<dictionary.size(); i++) {
    //    cout << dictionary[i].patt << " ";
    //    cout << dictionary[i].code << endl;
    //}

}


/*
 *      --- HUFFMAN private ---
 */

bool Dictionary::loe(Hnode* a, Hnode* b){
    return a->count > b->count;
}

void Dictionary::build_Huffman_tree(const unsigned int* occ){
    vector<Hnode*> q;
    for(unsigned short i=0; i< getSize(); i++){
        if(occ[i]>0){
            //cout<<(char) i<<" "<<occ[i]<<endl;
            auto* new_node=(Hnode*)malloc(sizeof(Hnode));
            new_node->c=(unsigned int)i;
            new_node->count=occ[i];
            new_node->left_child=nullptr;
            new_node->right_child=nullptr;
            q.push_back(new_node);
            sort(q.begin(), q.end(), loe);
        }
    }
    //cout<<"-----------------"<<endl;
    while(q.size()>1){
        Hnode* a;
        Hnode* b;
        a=q[q.size()-1];
        q.pop_back();
        b=q[q.size()-1];
        q.pop_back();
        auto* new_node=(Hnode*)malloc(sizeof(Hnode));
        new_node->c= '\0';
        new_node->count=a->count + b->count;
        new_node->left_child=a;
        new_node->right_child=b;
        q.push_back(new_node);
        sort(q.begin(), q.end(), Dictionary::loe);
    }
    Hnode* Hroot=q[0];
    write_Huffman(Hroot, "");
}

void Dictionary::write_Huffman(Hnode* n, string s) {
    if (n->c == '\0') {
        string t = s;
        write_Huffman(n->right_child, t.append(1, '1'));
        write_Huffman(n->left_child, s.append(1, '0'));
    } else {
        //out<<(int) n->c<<" "<<s<<endl;
        dictionary[n->c].bin_len = s.size();
        for (int i = 0; i < s.size(); i++)
            dictionary[n->c].bin[i] = s[i];
        dictionary[n->c].bin[s.size()] = '\0';
    }

}

string Dictionary::getBinCode(unsigned int code_REq) {
    return string(dictionary[code_REq].bin);
}
