#cmake_minimum_required(VERSION 3.10)
#project(SMILESCompression)

#set(CMAKE_CXX_STANDARD 14)

#add_executable(generate_dictionaries pattern.cpp  utils.h Text.cpp Text.h HashTable.cpp HashTable.h Dictionary.cpp Dictionary.h utils.cpp Preprocess.cpp Preprocess.h Codec.cpp Codec.h)
#add_executable(pipeline main.cpp Dictionary.h Dictionary.cpp utils.h utils.cpp Preprocess.cpp Preprocess.h Codec.cpp Codec.h)
#add_executable(testing1 testing.cpp Dictionary.cpp Dictionary.h utils.cpp utils.h Preprocess.cpp Preprocess.h Codec.cpp Codec.h)

cmake_minimum_required(VERSION 3.22.1)

project(CUDA-ZSmile CUDA)

set(CMAKE_CUDA_COMPILER /opt/install/cuda/11.7/bin/nvcc)
include_directories(/opt/install/cuda/11.7/include)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CUDA_ARCHITECTURES 80)

set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb

#set (CMAKE_BUILD_TOOL Debug)
set (CMAKE_BUILD_TYPE Release)

#set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(pipeline main.cu Dictionary.cuh Dictionary.cu utils.cuh utils.cu Preprocess.cu Preprocess.cuh Codec.cu Codec.cuh)

set_target_properties(pipeline PROPERTIES CUDA_SEPARABLE_COMPILATION ON)