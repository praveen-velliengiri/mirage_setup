#!/bin/bash

# Compiler and standard
gpp=g++
std="-std=c++17"

# Include directories
include_dirs=(
    "-I/home/pravell/work/install/include"
    "-I/home/pravell/work/mirage/deps/cutlass/include"
    "-I/usr/local/cuda-12.1/include"
    "-I/home/pravell/work/mirage/deps/json/include"
    "-I/home/pravell/work/mirage/deps/z3/src/api"
    "-I/home/pravell/work/mirage/deps/z3/src/api/c++"
)

# Library directories
lib_dirs=(
    "-L/home/pravell/work/install/lib"
    "-L/usr/local/cuda-12.1/lib64/"
    "-L/home/pravell/work/mirage/deps/z3/build"
)

# Libraries
libs=(
    "-lmirage_runtime"
    "-lcudart"
    "-lcublas"
    "-lz3"
    "-lm"
)

# Source file
src="$1"

# Output executable
output="rms"

# Compile command
$gpp $std -fopenmp "${include_dirs[@]}" "$src" "${lib_dirs[@]}" "${libs[@]}" -o "$output"

# Make the output executable
chmod +x "$output"

echo "Compilation finished: ./$output"