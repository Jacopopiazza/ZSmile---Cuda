#include <cstring>
#include "Codec.cuh"
#include "utils.cuh"

Codec::Codec(const std::string& dictionary_filename, bool preprocess, bool permissive, bool huffman){
    dictionary = new Dictionary();
    dictionary->load_dictionary(dictionary_filename.c_str());
    int dict_size = dictionary->getSize();
    length = dictionary->get_length();
    length[ESCAPE] = 1;
    compression_tree = build_tree();
    if(huffman){
        for(int i=0; i<dict_size; i++)
            huffman_code[i]=dictionary->getBinCode(i);
        weight = dictionary->get_bin_length();
    }
    else{
        //not huffman compressor support just 256 wide dictionary
        weight = (int*)malloc(256 * sizeof(int));
        for(int i=0; i<256; i++) weight[i]=1;
        weight[ESCAPE]=2;
    }

    table=dictionary->print_table();
    if(huffman){
        huffman_tree = build_dec_tree();
        magic_generate();
    }

    if(preprocess)
        p= new Preprocess(permissive);

    this->huffman = huffman;
    this->preprocess = preprocess;
}

Codec* Codec::cudaCodec(const std::string& dictionary_filename, bool preprocess, bool permissive, bool huffman){
    
    Codec* codec;
    cudaMallocManaged(&codec, sizeof(Codec));
    
    codec->dictionary = Dictionary::cudaDictionary();
    codec->dictionary->load_dictionary(dictionary_filename.c_str());
    int dict_size = codec->dictionary->getSize();
    codec->length = codec->dictionary->get_length();
    codec->length[ESCAPE] = 1;
    codec->compression_tree = codec->build_tree();
    if(codec->huffman){
        for(int i=0; i<dict_size; i++)
            codec->huffman_code[i]=codec->dictionary->getBinCode(i);
        codec->weight = codec->dictionary->get_bin_length();
    }
    else{
        //not huffman compressor support just 256 wide dictionary
        cudaMallocManaged(&(codec->weight), 256 * sizeof(int));

        for(int i=0; i<256; i++) codec->weight[i]=1;
        codec->weight[ESCAPE]=2;
    }

    codec->table=codec->dictionary->print_table();
    if(codec->huffman){
        codec->huffman_tree = codec->build_dec_tree();
        codec->magic_generate();
    }

    if(preprocess)
        codec->p = Preprocess::cudaPreprocess(permissive);

    codec->huffman = huffman;
    codec->preprocess = preprocess;

    return codec;
}



Codec::~Codec() {

    delete_tree(compression_tree);

    cudaFree(length);
    cudaFree(weight);

    cudaFree(table);
    
    
    if(huffman)
        delete_dec_tree(huffman_tree);

    if(preprocess)
        delete p;

    delete dictionary;
}

int Codec::getMaxLengthOfDictionaryPattern(){
    return this->dictionary->get_max_length();
}

int Codec::smile_compress(char* smile_in, char* smile_zip,int *scoreMatrix, unsigned int *bestMatrix,unsigned int *all_matchMatrix, char *sMatrix){
    //smile_zip.clear();
    smile_zip[0] = '\0';

    char *s = sMatrix;
    unsigned int *all_match = all_matchMatrix;
    unsigned int *best = bestMatrix;
    int *score = scoreMatrix;


    int size;

    if(preprocess){
        if(p->preprocess_ring(smile_in, s)!=0){
            return -1;
        }

        size=myStrlen(s);
    } else{

        size=myStrlen(smile_in);

        for(int i=0; i<size; i++){
            s[i]=smile_in[i];
        }
        s[size]='\0';
    }

    for(int i=0; i< size; i++){
        int j=0;
        node* curr = compression_tree;
        int len =0;
        while(i +len < size && curr!= nullptr){
            curr = curr -> next[s[i+len]];
            if(curr== nullptr) break;
            if(curr->isValid){
                all_match[i*MAX_PATT_LEN + j]=curr->code;
                j++;
            }
            len++;
        }
        //for(; j<MAX_PATT_LEN; j++)
        all_match[i*MAX_PATT_LEN + j]='\0';
    }


    for(int i= size-1; i>=0; i--){
        int j=0;
        score[i]=2*size+1;
        if(huffman) score[i]*=16;
        while(j<MAX_PATT_LEN && all_match[i*MAX_PATT_LEN + j]!='\0'){
            unsigned int pattern = all_match[i*MAX_PATT_LEN + j];
            int next = i + length[all_match[i*MAX_PATT_LEN + j]];
            if(next >= size){
                score[i]=weight[pattern];
                best[i]=pattern;
            }
            else if(score[i] > weight[pattern] + score[next]){
                score[i]= weight[pattern] + score[next];
                best[i]=pattern;
            }
            j++;
        }
    }

    if(huffman){
        unsigned char buffer ='\0';
        int count = 0;
        for (int i = 0; i < size;) {
            for(char bit: huffman_code[best[i]]){
                buffer = buffer<<1;
                if(bit == '1') buffer++;
                count++;
                if(count%8==0){
                    cudaPushBack(smile_zip, buffer);
                    buffer='\0';
                }
            }
            i += length[best[i]];
        }
        if(count%8>0){
            buffer = buffer << (8-count%8);
            buffer += magic_sequence[8-count%8];
            cudaPushBack(smile_zip, buffer);

        }
        //smile_zip[0]=count/256;
        //smile_zip[1]=count%256;
    }else {
        for (int i = 0; i < size;) {
            cudaPushBack(smile_zip, best[i]);
            if(best[i]==ESCAPE){
                cudaPushBack(smile_zip, s[i]);
            }
            i += length[best[i]];
        }
    }

    return 0;
}

//int Codec::smile_decompress(char* smile_zip, char* smile_out){
int Codec::smile_decompress(char* smile_zip, char* smile_out){
    if(!huffman){
        return smile_decompress_std(smile_zip, smile_out);
    }else{
        return smile_decompress_huffman(smile_zip, smile_out);
    }
}


int Codec::smile_decompress_std(char* smile_zip, char* smile_out){

    int j=0;
    char s_out[MAX_SENTENCE_LEN];

    int len;
    len  = myStrlen(smile_zip);

    for(int index =0; index <len; index++){
        if(smile_zip[index]!=ESCAPE){
            int k=0;
            while(table[((unsigned char)smile_zip[index])*MAX_PATT_LEN +k]!='\0') {
                s_out[j] = table[((unsigned char) smile_zip[index]) * MAX_PATT_LEN + k];
                k++;
                j++;
            }
        }else{
            index++;
            s_out[j] = smile_zip[index];
            j++;
        }
    }
    int size=j;
    s_out[j] = '\0';
    if(preprocess){
        p->postprocess_ring(s_out, size, smile_out);
    }else{
        smile_out[0] = '\0';
        myStrcpy(smile_out, s_out);    
    }

    return 0;
}

int Codec::smile_decompress_huffman(char* smile_zip, char* smile_out){

    int j=0;
    char s_out[MAX_SENTENCE_LEN];

    int zip_size = strlen(smile_zip)*8;
    
    unsigned char temp = smile_zip[0];
    int i=0;
    while(i<zip_size){
        decNode* curr= huffman_tree;
        while(i<zip_size && curr->c=='\0'){
            if(temp/128==0){
                curr = curr->left;
            }else{
                curr = curr->right;
            }
            temp = temp<<1;
            i++;
            if(i%8==0){
                temp = smile_zip[(i/8)];
            }
        }
        if(curr->c != '\0') {
            unsigned int code = curr->c;
            int k = 0;
            while (table[code * MAX_PATT_LEN + k] != '\0') {
                s_out[j] = table[code * MAX_PATT_LEN + k];
                k++;
                j++;
            }
        }
    }
    int final_size=j;
    //for (; j < MAX_SENTENCE_LEN; j++)
    s_out[j] = '\0';
    if(preprocess){
        p->postprocess_ring(s_out, final_size, smile_out);
    }else{

        smile_out[0] = '\0';
        myStrcpy(smile_out, s_out);

    }

    return 0;

}


decNode *Codec::build_dec_tree() {
    auto* root = (decNode*)malloc(sizeof(decNode));
    root->c='\0';
    root->left = nullptr;
    root->right = nullptr;

    for(int i=0; i<dictionary->getSize(); i++){
        std::string temp = dictionary->getBinCode(i);
        decNode* curr = root;
        for(char c: temp){
            if(c=='0'){
                if(curr->left== nullptr){
                    curr->left = (decNode*)malloc(sizeof(decNode));
                    curr->left->c='\0';
                    curr->left->left= nullptr;
                    curr->left->right = nullptr;
                }
                curr= curr->left;
            }
            if(c=='1'){
                if(curr->right== nullptr){
                    curr->right = (decNode*)malloc(sizeof(decNode));
                    curr->right->c='\0';
                    curr->right->left= nullptr;
                    curr->right->right = nullptr;
                }
                curr= curr->right;
            }
        }
        curr->c = i;
    }
    root->c='\0';
    return root;
}

node* Codec::build_tree(){

    node* root;//=(node*) calloc(1, sizeof(node));
    cudaMallocManaged(&(root), sizeof(node*));
    for(short i=1; i<PRINTABLE; i++){
        //root->next[i]= (node*) calloc(1, sizeof(node));
        cudaMallocManaged(&(root->next[i]), sizeof(node));
        root->next[i]->isValid=true;
        root->next[i]->code = (unsigned char)i;
    }
    pattern p;
    unsigned char* curr;
    node* pos;
    for(unsigned int i=FIRST_CHAR; i<dictionary->getSize(); i++){
        p = dictionary->getPatt(i);
        curr = p.patt;
        if(curr[0]!='\0') {
            if(i<PRINTABLE && curr[1]!='\0'){
                root->next[i]->code = ESCAPE;
            }
            pos = root;
            while (*curr != '\0') {
                if (pos->next[*curr] == nullptr) {
                    //pos->next[*curr] = (node *) calloc(1, sizeof(node));
                    cudaMallocManaged(&(pos->next[*curr]), sizeof(node));
                }
                pos = pos->next[*curr];
                curr++;
            }
            pos->isValid = true;
            pos->code = i;
        }
    }

    //find_error(root, root, "");
    return root;
}

void Codec::delete_tree(node* n){
    for(int i=0; i<PRINTABLE; i++){
        if(n->next[i]!= nullptr){
            delete_tree(n->next[i]);
        }
    }
    cudaFree(n);
}

void Codec::delete_dec_tree(decNode* root){
    if(root->left!= nullptr)
        delete_dec_tree(root->left);
    if(root->right != nullptr)
        delete_dec_tree(root->right);


    cudaFree(root);
}

void Codec::magic_generate() {
    magic_sequence[0]=0;    //useless
    int divider = 1;
    for(int i=1; i<8; i++){
        bool valid=false;
        unsigned char magic=0;
        while(!valid){
            valid=true;
            unsigned char temp = magic;
            decNode * curr = huffman_tree;
            for(int j=0; j<i && valid; j++){
                if(curr== nullptr)
                    valid=false;
                else{
                    if(temp/divider==0){
                        curr= curr->left;
                    }else{
                        curr = curr->right;
                    }
                }
                temp = temp % divider;
                temp = temp << 1;
            }
            if(curr==nullptr || curr->c!='\0'){
                valid=false;
                magic++;
            }
        }
        divider*=2;
        magic_sequence[i]=magic;
    }
}




