#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include "utils.cuh"
#include "Codec.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: Error Name:%s,ErrorString:%s, %s %d\n", cudaGetErrorName(code), cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int readSmiles(std::ifstream &input_file, char* data, unsigned int *indexes, unsigned int &numSmiles, int max_smiles, unsigned long long int &totalReadByte){

    std::string input;

    numSmiles = 0;
    int lastIndex = 0;
    totalReadByte = 0;
    int maxLen = -1;

    while (numSmiles < max_smiles && std::getline(input_file, input) && !input.empty()) {
        
        // Process the input
        //Get len of input at most MAX_SENTENCE_LEN
        int len = std::min((int)input.length(), MAX_SENTENCE_LEN - 1);

        //Copy input to data
        memcpy(data + lastIndex, input.c_str(), len);
        //Add null terminator
        data[lastIndex + len] = '\0';

        //add sum to total byte
        totalReadByte += len + 1;

        if(maxLen < len + 1){
            maxLen = len + 1;
        }

        //Update indexes
        indexes[numSmiles] = lastIndex;
        //Update lastIndex
        lastIndex += len + 1;
        //Update numSmiles
        numSmiles++;

    }

    return maxLen;


}



__global__ void decompressKernel(char* data, unsigned int *indexes, char* output, int numSmiles, int max_smiles, int step, Codec* codec){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < numSmiles; i += stride){
        int smile_idx = indexes[i];
        unsigned int increment = i * step;
        codec->smile_decompress(data + smile_idx, output + increment);

    }


}

__global__ void compressKernel(char* data, unsigned int *indexes, char* output, int numSmiles, int max_smiles, int step, Codec* codec,int *scoreMatrix, unsigned int *bestMatrix,unsigned int *all_matchMatrix, char *sMatrix){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;


    for(int i = idx; i < numSmiles; i += stride){
        int smile_idx = indexes[i];
        unsigned int increment = i * step;

        codec->smile_compress(data + smile_idx, output + increment, scoreMatrix + idx * MAX_SENTENCE_LEN, bestMatrix + idx * MAX_SENTENCE_LEN, all_matchMatrix + idx * MAX_SENTENCE_LEN * MAX_PATT_LEN, sMatrix + idx * MAX_SENTENCE_LEN);
    }


}



int main(int argc, const char * argv[]) {

    bool preprocess{false};
    bool verbose{false};
    bool decompress{false};
    bool compress{false};
    bool check{false};
    bool set_dictionary{false};
    bool set_input{false};
    bool set_output{false};
    bool permissive{false};
    bool huffman{false};
    bool throughput{false};
    bool print_size{false};
    bool csv{false};
    bool gpu{false};
    std::string dict_filename = "";
    std::string in_filename = "";
    std::string out_filename = "";

    //Process command-line arguments
    int arg;
    for(arg=1; arg < argc; arg++){
        std::string arg_str{argv[arg]};

        if (arg_str == "--dictionary")
            if (++arg < argc){
                dict_filename = std::string(argv[arg]);
                set_dictionary = true;
            }
            else{
                std::cerr << "Missing dictionary filename! Usage: --dictionary <dictionary_filename>" << std::endl;
                exit(1);
            }
        else if(arg_str == "--preprocess"){
            preprocess = true;
        }
        else if(arg_str == "--verbose"){
            verbose = true;
        }
        else if(arg_str == "--check"){
            check = true;
        }
        else if(arg_str == "--compress"){
            compress = true;
        }
        else if(arg_str == "--decompress"){
            decompress = true;
        }
        else if(arg_str == "--permissive"){
            permissive = true;
        }
        else if(arg_str == "--huffman"){
            huffman = true;
        }
        else if(arg_str == "--throughput"){
            throughput=true;
        }
        else if(arg_str == "--csv"){
            csv= true;
        }else if(arg_str == "--print_size"){
            print_size=true;
        }
        else if(arg_str == "--gpu"){
            gpu = true;
        }
        else if(arg_str == "--input"){
            if (++arg < argc){
                in_filename = std::string(argv[arg]);
                set_input = true;
            }
            else{
                std::cerr << "Missing input filename! Usage: --input <input_filename>" << std::endl;
                exit(1);
            }
        }
        else if(arg_str == "--output"){
            if (++arg < argc){
                out_filename = std::string(argv[arg]);
                set_output = true;
            }
            else{
                std::cerr << "Missing output filename! Usage: --output <output_filename>" << std::endl;
                exit(1);
            }
        }
        else {
            std::cerr << "Invalid arguments! " << std::endl;
            exit(1);
        }
    }
    if (!set_dictionary)
    {
        std::cerr << "Invalid arguments, missing dictionary file! Use --dictionary" << std::endl;
        exit(1);
    }
    if ((!compress)&&(!decompress)&&(!check))
    {
        std::cerr << "Invalid arguments, missing compress/decompress action! Use --compress or --decompress or --check" << std::endl;
        exit(1);
    }
    if(!set_input || !set_output){
        std::cerr << "Invalid arguments, missing input/output file! Use --input and --output" << std::endl;
        exit(1);
    }
    if(gpu && huffman){
        std::cerr << "Huffman compression and decompression is not supported on GPUs" << std::endl;
        exit(1);
    }
    
    /* CALCULATING CONSTANT VALUES */
    std::ifstream input_file;
    input_file.open(in_filename, std::ios::in );
    if (!input_file.is_open()) {
        std::cerr << "Error opening intput file!" << std::endl;
        exit(1);
    }
    std::ofstream output_file;
    output_file.open(out_filename, std::ios::out );
    if (!output_file.is_open()) {
        std::cerr << "Error opening output file!" << std::endl;
        exit(1);
    }
    

    if(gpu){

        int device;
        cudaGetDevice(&device);
        //gpuErrchk( cudaPeekAtLastError() );

        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, device);
        //gpuErrchk( cudaPeekAtLastError() );

        Codec* codec = Codec::cudaCodec(dict_filename, preprocess, permissive, huffman);
        //gpuErrchk( cudaPeekAtLastError() );


        //Parameters for streams
        const int warpSize = devProp.warpSize;

        int numStreams = 8;
        cudaStream_t streams[numStreams];
        for(int i = 0; i < numStreams; i++){
            cudaStreamCreate(&streams[i]);
        }

        //Constant and pointers for kernels
        const int blockBatch = 4;
        const int threadBatch = 16 * warpSize;
        const int totalThread = blockBatch * threadBatch;
        const int totalSmiles = 2 * totalThread;

        char* data, *data_gpu, *output_gpu, *output_cpu;
        unsigned int *indexes, *indexes_gpu, *bestMatrix, *all_matchMatrix;
        int *scoreMatrix;
        char *sMatrix;

        //Lengths to access array as if they were matrix
        const int lengthScoreMatrix = MAX_SENTENCE_LEN * totalThread;
        const int lengthBestMatrix =  MAX_SENTENCE_LEN * totalThread;
        const int lengthAllMatchMatrix = MAX_SENTENCE_LEN * MAX_PATT_LEN * totalThread;
        const int lengthSMatrix =  MAX_SENTENCE_LEN * totalThread;
        
        //Allocate those structures only when compressing
        if(compress){
            cudaMalloc(&scoreMatrix, numStreams * sizeof(int) * lengthScoreMatrix);
            cudaMalloc(&bestMatrix, numStreams * sizeof(unsigned int) * lengthBestMatrix);
            cudaMalloc(&all_matchMatrix, numStreams * sizeof(unsigned int) * lengthAllMatchMatrix);
            cudaMalloc(&sMatrix, numStreams * lengthSMatrix * sizeof(char));
            //gpuErrchk( cudaPeekAtLastError() );
        }

        //Allocate memory on host
        const int lengthData = totalSmiles * MAX_SENTENCE_LEN;
        const int lengthIndexes =  totalSmiles;
        const int lengthOutput = totalSmiles * MAX_SENTENCE_LEN;

        //Using cudaMallocHost to be able to copy those to device
        cudaMallocHost(&data, sizeof(char) * numStreams * lengthData);
        //gpuErrchk( cudaPeekAtLastError() );

        cudaMallocHost(&indexes, sizeof(unsigned int) * numStreams * lengthIndexes);
        //gpuErrchk( cudaPeekAtLastError() );

        cudaMallocHost(&output_cpu,  sizeof(char) * numStreams * lengthOutput);
        //gpuErrchk( cudaPeekAtLastError() );

        //Generic data needed for both compressiong and decompression
        cudaMalloc(&data_gpu, sizeof(char) * numStreams * lengthData);
        //gpuErrchk( cudaPeekAtLastError() );

        cudaMalloc(&indexes_gpu, sizeof(unsigned int) *numStreams * lengthIndexes);
        //gpuErrchk( cudaPeekAtLastError() );

        cudaMalloc(&output_gpu,  sizeof(char) * numStreams * lengthOutput);
        //gpuErrchk( cudaPeekAtLastError() );
        
        //Long vars to store amount of processed bytes
        unsigned long long int ltot_smile_in=0;
        unsigned long long int totalReadByte[numStreams] = {0};

        unsigned int numSmiles[numStreams] = {0};
        bool exit = false;
        int last_valid_stream = numStreams;
        unsigned long long processed= 0;
        int maxPatternLen = codec->getMaxLengthOfDictionaryPattern();


        //Max len per each stream, used to try to minize as mush as possibile DtoH data transfers
        int maxLen[numStreams] = {0};
        
        /* ACTUAL COMPRESSION / DECOMPRESSION */
        auto initial = std::chrono::high_resolution_clock::now();

        while (!exit) {
            
            //Read a batch of smile per stream
            for(int i = 0;i<numStreams;i++){
                maxLen[i] = 0;
                numSmiles[i] = 0;
                totalReadByte[i] = 0;
                maxLen[i] = readSmiles(input_file, data + i*lengthData, indexes + i*lengthIndexes, numSmiles[i], totalSmiles, totalReadByte[i]);
                
                if(numSmiles[i] == 0 || totalReadByte[i] == 0){
                    last_valid_stream = i;
                    exit = true;
                    break;
                }
            }

            //Launch async HtoD copies of both smiles and their indexes
            for(int i= 0;i<last_valid_stream;i++){
                ltot_smile_in += totalReadByte[i];
                cudaMemcpyAsync(data_gpu + i*lengthData, data + i*lengthData, totalReadByte[i], cudaMemcpyHostToDevice,streams[i]);
                //gpuErrchk( cudaPeekAtLastError() );

                cudaMemcpyAsync(indexes_gpu + i*lengthIndexes, indexes + i*lengthIndexes, numSmiles[i] * sizeof(unsigned int), cudaMemcpyHostToDevice, streams[i]);
                //gpuErrchk( cudaPeekAtLastError() );
            }
            
            //Launch kernels
            for(int i = 0;i<last_valid_stream;i++){
                if (compress){
                    //When compressing ZSmiles cannot be greater than the input smiles

                    compressKernel<<<blockBatch, threadBatch, 0, streams[i]>>>(data_gpu + i*lengthData, indexes_gpu + i*lengthIndexes, output_gpu+i*lengthOutput, numSmiles[i], totalSmiles, maxLen[i], codec ,scoreMatrix+i*lengthScoreMatrix, bestMatrix+i*lengthBestMatrix, all_matchMatrix+i*lengthAllMatchMatrix, sMatrix+i*lengthSMatrix);
                    //gpuErrchk( cudaPeekAtLastError() );
                }
                else if (decompress){
                    //Only when decompressing, we recalculate the max possible length to support even the worst case
                    //Worst case: when a zsmile is a sequence of chars to be replaced with the longest pattern
                    maxLen[i] = std::min(maxLen[i]*maxPatternLen+1, MAX_SENTENCE_LEN);

                    decompressKernel<<<blockBatch, threadBatch, 0, streams[i]>>>(data_gpu + i*lengthData, indexes_gpu+i*lengthIndexes, output_gpu+i*lengthOutput, numSmiles[i], totalSmiles, maxLen[i], codec);
                    //gpuErrchk( cudaPeekAtLastError() );
                }

            }

            //Launch DtoH copuies using calculated maxLen
            for(int i = 0;i<last_valid_stream;i++){
                cudaMemcpyAsync(output_cpu +i*lengthOutput, output_gpu + i*lengthOutput, sizeof(char) * numSmiles[i] * maxLen[i] , cudaMemcpyDeviceToHost,streams[i]);
                //gpuErrchk( cudaPeekAtLastError() );
                processed += numSmiles[i];
            }
            
            //Wait for all operations to end
            cudaDeviceSynchronize();
            //gpuErrchk( cudaPeekAtLastError() );

            char *pos;
            //Print smiles on output file
            for(int j = 0;j<last_valid_stream;j++){
                for(int i = 0; i < numSmiles[j]; i++){
                    pos = output_cpu +j*lengthOutput + i* maxLen[j];
                    output_file << pos << "\n";
                }
            }
            
            if(exit){
                break;
            }

        }

        //If stats were requested print them
        if(throughput){
            double time = (double) std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -  initial).count() / 1000.0;
            double mega=  (double)ltot_smile_in/(1024.0*1024.0);
            std::cout <<" Throughput = "<< (mega)/(time)<<" MB/s\n";
            std::cout << "Processed = " << ltot_smile_in  << " Byte\n";
        }

        //Destroy streams
        for(int i = 0;i<numStreams;i++){
            cudaStreamDestroy(streams[i]);
        }


    }
    else{

        char* smile_in_c;
        char* smile_out_c;
        char* smile_zip_c;

        smile_in_c = (char *)calloc(MAX_SENTENCE_LEN, sizeof(char));
        smile_out_c = (char *)calloc(MAX_SENTENCE_LEN, sizeof(char));
        smile_zip_c = (char *)calloc(MAX_SENTENCE_LEN, sizeof(char));
        unsigned long long int ltot_smile_in=0, ltot_smile_zip=0;

        Codec codec(dict_filename, preprocess, permissive, huffman);
        //Codec* codec = Codec::cudaCodec(dict_filename, preprocess, permissive, huffman);
        int score[MAX_SENTENCE_LEN];
        unsigned int best[MAX_SENTENCE_LEN];
        unsigned int all_match[MAX_SENTENCE_LEN*MAX_PATT_LEN];
        char s[MAX_SENTENCE_LEN];

        /* ACTUAL COMPRESSION/DECOMPRESSION */
        auto initial = std::chrono::high_resolution_clock::now();
        while(input_file.getline(smile_in_c, MAX_SENTENCE_LEN)){

            if (compress){
                if(codec.smile_compress(smile_in_c, smile_zip_c, score, best, all_match, s)==0) {
                    output_file << smile_zip_c << std::endl;
                }else{
                    output_file<<"INVALID SMILE"<<std::endl;
                }
            }
            else if (decompress){
                codec.smile_decompress(smile_in_c, smile_out_c);
                output_file<< smile_out_c <<std::endl;
            }
            else {  /* check */
                if(codec.smile_compress(smile_in_c, smile_zip_c,score,best,all_match,s)==0){
                    codec.smile_decompress(smile_zip_c, smile_out_c);
                }else{
                    output_file<<"INVALID SMILE"<<std::endl;
                    output_file<< smile_in_c <<std::endl;
                    strcpy(smile_out_c, smile_in_c);
                }
                if(verbose){
                    output_file<< smile_in_c <<std::endl;
                }
                if (strcmp(smile_in_c,smile_out_c) != 0){
                    output_file<<"SMILE DIFFERS"<<std::endl;
                    output_file<<smile_in_c<<std::endl;
                    output_file<<smile_out_c<<std::endl;
                }
            }

            if (verbose&&compress){
                int l_smile_in = strlen(smile_in_c);
                int l_smile_zip = strlen(smile_zip_c);
                output_file<<" Smile in: " << smile_in_c <<std::endl;
                output_file<<" Smile out: " << smile_zip_c <<std::endl;
                output_file<<" Compression ratio = "<< ((float)l_smile_zip)/((float)l_smile_in)<< std::endl;
            }
            if(throughput || check){
                ltot_smile_in += strlen(smile_in_c);
                ltot_smile_zip += strlen(smile_zip_c);
            }
            
        }
        if(check){
            if(csv){
                if(print_size) output_file<<(((double)ltot_smile_in)/(1024.0*1024.0*1024.0))<<";";
                output_file<<((double)ltot_smile_zip)/((double)ltot_smile_in)<<";";
            }
            else output_file<<"Compression ratio = "<< ((double)ltot_smile_zip)/((double)ltot_smile_in)<< " Input Size = "<< (((double)ltot_smile_in)/(1024.0*1024.0*1024.0))<<std::endl;
        }
        
        if(throughput){
            double time = (double) std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -  initial).count() / 1000.0;
            double mega=  (double)ltot_smile_in/(1024.0*1024.0);
            std::cout<<" Throughput = "<< (mega)/(time)<<" MB/s" << std::endl;
        }
    }

   
    

    return 0;
}