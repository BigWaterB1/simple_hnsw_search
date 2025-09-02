#!/bin/bash

# 编译C++查询程序
echo "Compiling query_index.cpp..."

g++ -std=c++17 -I./hnswlib query_index.cpp -o query_index

echo "Compilation complete. Run with: ./query_index"