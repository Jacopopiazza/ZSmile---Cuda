#include "utils.cuh"
#include "HashTable.cuh"
#include "Dictionary.cuh"
#include "Text.cuh"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
//OOP version
using namespace std;

void find_dictionary(bool preprocess, int size, string &alphabet){

    auto* hashTable = new HashTable();
    auto* dictionary = new Dictionary();
    if(alphabet=="")
        dictionary->load_alphabet();
    else
        dictionary->load_alphabet(alphabet.c_str());
    if(!dictionary->set_size(size)) return;
    //Text * text = new Text(training_file);
    Text * text = new Text(preprocess);


    //new version
    text->find_all_patterns(MIN_PATT_LEN, MAX_PATT_LEN, hashTable);

    cout<<hashTable->getUsage()<<endl;
    while (! dictionary->isFull()){
        //get new dictionary entry
        patt_counter best = hashTable->getMax();
        unsigned int code = dictionary->getCode();
        cout<< best.patt <<" "<< best.count<<" "<<code<<endl;

        //update dictionary
        dictionary->add(best);

        //update text
        text->update(best, code, hashTable, dictionary);
        if(dictionary->getTotalScore()!=text->getSavedSize()){
            cout<<"errore"<<endl;
        }
        if(!hashTable -> safe_delete()){
            cout<<"error"<<endl;
        }


    }

/*
    int loop=0;
    //improved search
    text->find_all_patterns(MIN_PATT_LEN, MAX_PATT_LEN, hashTable);
    while(!dictionary->isFull() || dictionary->getLower() < hashTable ->getMax().count){

        if(loop>100) return;
        if(dictionary->getLower()>0 && dictionary->getLower() < hashTable ->getMax().count) {
            cout<<"Something new"<<endl;
            loop++;
            //remove older
            unsigned int toDelete = dictionary->getLowerCode();
            short toDeleteLength = dictionary->getPattLen(toDelete);

            text->remove(toDelete, toDeleteLength, hashTable);
            dictionary->removeCode(toDelete);
        } else{loop=0;}

        //get new dictionary entry
        patt_counter best = hashTable->getMax();
        unsigned int code = dictionary->getCode();
        //cout<< best.patt <<" "<< best.count<<" "<<code<<endl;

        //update dictionary
        dictionary->add(best);

        //update text
        text->update(best, code, hashTable, dictionary);
        if(dictionary->getTotalScore()!=text->getSavedSize()){
            cout<<"errore"<<endl;
        }
        if(!hashTable -> safe_delete()){
            cout<<"error"<<endl;
        }

    }*/

    /*
    //branch and bound
    //TODO not complete
    //TODO need to find a smarter way to store hashtable, it use 2Gb, it's to much even for a DFS strategy
    text->find_all_patterns(MIN_PATT_LEN, MAX_PATT_LEN, hashTable);
    auto* test = hashTable->getMax(10);
    for(int i=0; i<10; i++){
        cout<<test[i].patt<<" "<<test[i].count<<endl;
    }

    */

    //console output, NOT strictly necessary
    float c=text->getRatio();
    cout<< c<<endl;



    //common part --- DO NOT comment the following statements
    //print encoded file
    //text-> print("dict_encoded.txt");


    //Huffman encoding
    dictionary->compute_Huffman(text->calculate_frequencies());


    //write dictionary file
    //dictionary->print_to_file(dictionary_file);
    dictionary->print_to_std_out();

    //log csv format
    //*log<<text->getOriginalSize()<<";";
    //*log<<text->getRatio()<<";";

    //destructor
    delete hashTable;
    delete text;
    delete dictionary;
}



/*
 *
 * Function to generate test files
 *
 */

void generate_test_file( int n, const char * input_file, const char * test_file){
    ifstream in(input_file);
    ofstream out(test_file);

    srand(time(0));
    int skip;
    string s;

    for(int i=0; i< n; i++){
        skip = rand() % 1000;
        for(int j=0; j<skip; j++){
            if(in.eof()) {
                in.close();
                in.open(input_file);
            }
            getline(in, s);
        }
        out<<s;
        if(i < n-1) out<<endl;
    }

}

void generate_test_file_index(int n, int mod, const char * index_file, const char * test_file){
    ifstream in(index_file);
    vector<string> index;
    string s;
    while(!in.eof()){
        getline(in, s);
        index.push_back(s);
    }

    int index_t=0;
    in.open(index[index_t]);
    ofstream out(test_file);

    srand(time(0));
    int skip;

    for(int i=0; i< n; i++){
        skip = rand() % mod;
        for(int j=0; j<skip; j++){
            if(in.eof()) {
                in.close();
                index_t++;
                index_t%=index.size();
                in.open(index[index_t]);
            }
            getline(in, s);
        }
        out<<s;
        if(i < n-1) out<<endl;
    }

}


int main(int argc, const char * argv[]) {

    bool preprocess{false};
    bool alphabet{false};
    std::string alphabet_file="";
    int size = 256;

    int arg;
    for(arg=1; arg < argc; arg++) {
        std::string arg_str{argv[arg]};
        if (arg_str == "--size")
            if (++arg < argc){
                size = std::stoi(std::string(argv[arg]));
            }
            else{
                std::cerr << "Missing dictionary size! Usage: --size <dictionary_size>" << std::endl;
                exit(1);
            }
        else if(arg_str == "--alphabet"){
            if (++arg < argc){
                alphabet_file= std::string(argv[arg]);
            }
            else{
                std::cerr << "Missing dictionary size! Usage: --size <dictionary_size>" << std::endl;
                exit(1);
            }
        }
        else if (arg_str == "--preprocess") {
            preprocess = true;
        } else {
            std::cerr << "Invalid arguments! " << std::endl;
            exit(1);
        }
    }


    //generate dictionaries
    find_dictionary(preprocess, size, alphabet_file);

    //generate_test_file_index(500000, 10000,"index.txt", "500k_lib.smi");

    return 0;
}
