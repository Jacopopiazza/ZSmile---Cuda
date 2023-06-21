#include <cstdlib>
#include <string.h>
#include <string>
#include <fstream>
#include <iostream>

using namespace std;

char* from_string(string s){
    char* res= (char*)malloc(128*sizeof(char));
    for(int i=0; i<s.size(); i++){
        res[i]=s[i];
    }
    res[s.size()]='\0';
    return res;
}

__host__ __device__ void pushBack(char* string, char c){
    int len = strlen(string);
    string[len] = c;
    string[len+1] = '\0';
}



__device__ __host__ int myStrlen(char* string){
    int i=0;
    while(string[i]!='\0'){
        i++;
    }
    return i;
}

__device__ __host__ void myStrcpy(char* dest, const char* source){
    int i=0;
    while(source[i]!='\0'){
        dest[i]=source[i];
        i++;
    }
    dest[i]='\0';
}

__device__ __host__ void cudaPushBack(char* string, char c){
    int len = myStrlen(string);
    string[len] = c;
    string[len+1] = '\0';
}

bool diff(const char * first_file, const char * second_file){
    ifstream a(first_file);
    ifstream b(second_file);

    bool res=true;
    int i=0;
    string s1, s2;
    while(res && !a.eof() && !b.eof()){
        getline(a, s1);
        getline(b, s2);
        if(s1!=s2){
            res=false;
            cout<<"FIRST MISMATCH AT ROW: "<<i<<endl;
        }
        i++;
    }

    return res;
}

short getLength(unsigned char* s) {
    short length = 0;              //pattern length (2-15)
    while (s[length] != '\0') length++;
    return length;
}

/*
* NO MORE USED FUNCTIONS
*

void stat(const char * file);
void clear_file(const char * file);
void clear_lib(const char * file);
void create_alphabet();
void generate_test_file( int n, const char * input_file, const char * test_file);
void generate_test_file_index(int n, int mod, const char * index_file, const char * test_file);

bool loe (var a, var b){
    if (a.val!=b.val)
        return a.val>b.val;
    return a.let<b.let;
}

void stat(const char * file) {
    ifstream in (file);
    ofstream out ("stat.txt");

    vector <var> car(256);
    //vector <float> perc(256);
    for(int i=0; i<256; i++){
        car[i].val=0;
        car[i].let = (char) i;
    }

    string s;

    while(!in.eof()){
        getline(in, s);
        for(int i=0; i<s.size(); i++){
            car[s[i]].val++;
        }

    }

    sort(car.begin(), car.end(), loe);
    long long int tot=0;
    for(int i=0; i<256; i++) tot+=car[i].val;

    for(int i=0; i<256; i++) car[i].perc = (float)car[i].val * 100.0 / (float) tot;

    for(int i=0; i<256; i++) {
        if (car[i].val > 0)
            out << "'" <<car[i].let << "', ";
    }

    out<<endl;
    for(int i=0; i<256; i++) {
        if (car[i].val > 0)
            out << car[i].val << ", ";
    }
}

void clear_file(const char * file){
    ifstream in(file);
    ofstream out("cleaned.txt");
    string s;

    while(!in.eof()){
        getline(in, s);
        if(s.size()>0) {
            if (s[s.size() - 1] == '\r') {
                s[s.size() - 1] = '\n';
                out << s;
            } else {
                out << s;
            }
        }
    }
}

void clear_lib(const char * file){
    ifstream in(file);
    ofstream out(&file [ '.smi']);
    string s;

    getline(in, s);
    while(!in.eof()){
        getline(in, s);
        int i=0;
        while(s[i]!='\t'){
            out<<s[i];
            i++;
        }
        if(!in.eof()) out<<endl;
    }
}

void create_alphabet(){
    bool used_char[256];
    for (bool &i : used_char) i = false;

    //compute used_char vector
    ifstream index("index.txt");
    ifstream in;
    string file_name;
    string s;
    while (!index.eof()) {
        getline(index, file_name);
        cout << file_name << endl;
        char c;
        in.open(from_string(file_name));
        in.get(c);
        while (!in.eof()) {
            used_char[c] = true;
            in.get(c);
        }
        in.close();
    }

    ofstream log("alphabet.txt");
    for (int i = 0; i < LAST_CHAR; i++) {
        if (used_char[i])
            log << i << " " << (char) i << endl;
    }
}
*/

/*
void find_error(node* root, node* pos, string s){
    if(pos->isValid) {
        pos->error = root;
        pos->offset = s.size();
    }
    else{
        for(short offset=1; offset<s.size(); offset++){
            node* exp=root;
            short index=offset;
            bool good=true;
            while(good && index<s.size()){
                if(exp->next[s[index]]!= nullptr){
                    exp=exp->next[s[index]];
                    index++;
                }
                else{
                    good=false;
                }
            }
            if(good){
                pos->offset=offset;
                pos->error=exp;
                offset=s.size();
            }
        }
    }

    for(short i=0; i<256; i++){
        if(pos->next[i]!=nullptr){
            string temp=s;
            find_error(root, pos->next[i], temp.append(1, i));
        }
    }
}
*/
