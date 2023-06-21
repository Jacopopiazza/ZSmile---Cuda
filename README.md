## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Utilization](#utilization)

## General info
This project is a CUDA implementation of the lossless online codec/decode technique for [SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) strings ZSmile.
The key point of this implementation is to try to accelerate as much as possibile the existing single-threaded application, exploiting the massive parallelism offered by Nvidia GPUs. 

This project is created in the context of the exam "Progetto di Ingegneria informatica" @ [Politecnico di Milano](https://www.polimi.it/)

	
## Technologies
Project is created with:
* CUDA C/C++
* Dictionary encoding
* Data preprocess of SMILES string
	
## Setup
To run this project, install it locally and compile using cmake.
CUDA Toolkit is needed to build the project, verify the path is correct in CMakeList file.

```
$ cd CUDA-ZSmile
$ mkdir build
$ cmake -S . -B build
$ cd build
$ make
$ ./pipeline
```

## Utilization

To compress/decompress a file you just need to run the ``pipeline`` executable and specify the correct options.
The program just take the input from stdin and print the output to stdout. Here some simple examples:

* Compression of a ``.smi`` file using gpu

```
$ ./build/pipeline --dictionary dictionaries/intel-jam/dictionary189_10_20.dct --compress --preprocess --gpu --input in.smi --output out.zsmi
```

* Decompression of a ``.zsmi`` file using gpu

```
$ ./build/pipeline --dictionary dictionaries/intel-jam/dictionary189_10_20.dct --decompress --preprocess --gpu --input in.zsmi --output out.smi
```

* Compression of a ``.smi`` file using cpu

```
$ ./build/pipeline --dictionary dictionaries/intel-jam/dictionary189_10_20.dct --compress --preprocess --input in.smi --output out.zsmi
```

* Decompression of a ``.zsmi`` file using cpu

```
$ ./build/pipeline --dictionary dictionaries/intel-jam/dictionary189_10_20.dct --decompress --preprocess --input in.zsmi --output out.smi
```

## Options

Here the complete lis of available programme options for ``pipeline``

* ``--dictionary filename.dct`` **mandatory** specify the dictionary you want to use to achieve your co/decompression

* ``--compress`` compress SMILES with the specified dictionary. Read from input file and print to output file

* ``--decompress`` decompress SMILES encoded using the same dictionary. Read from input file and print to output file

* ``--preprocess`` preprocess data before compress (or post-process them after decompression). You need to specify this option both during compression and decompression if you want to use it.

* ``--throughput`` at the end of the execution print the average throughput ration (in MB/s)

* ``--huffman`` co/decompress data using a binary notation. Encoded SMILES will NOT be human-readable. You need to specify this option both during compression and decompression if you want to use it. Can be use with ``--preprocess``. **Cannot be used with ``--gpu``.**

* ``--check`` check the compression ratio of a given sequence of SMILES with the specified dictionary. Read from input file and print to output file just the Ratio. **Cannot be used with ``--gpu``.**

* ``--input filename.ext`` **mandatory** set the path of the input file

* ``--output filename.ext`` **mandatory** set the path of the output file

* ``--gpu`` make use of CUDA framework to accelerate operations with GPUs

**NOTE** : you need to specify **ONE AND ONLY ONE** option between ``--compress`` , ``--decompress``, ``--check``
