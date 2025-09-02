#include "hnswlib/hnswlib.h"
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <sstream>
#include <memory>
#include <cstdio>
#include <algorithm>
#include <fstream>
#include <unordered_map>

std::vector<float> get_embedding_from_text(const std::string& text) {
    // 构建命令
    std::string command = "python embed.py '" + text + "'";
    
    // 执行Python命令并获取输出
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }

    // 读取输出
    std::string result;
    char buffer[128];
    while (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
        result += buffer;
    }

    // 解析向量（逗号分隔的浮点数）
    std::vector<float> embedding;
    std::istringstream iss(result);
    std::string token;
    
    while (std::getline(iss, token, ',')) {
        try {
            embedding.push_back(std::stof(token));
        } catch (const std::exception& e) {
            // 忽略转换错误
        }
    }

    return embedding;
}

std::unordered_map<hnswlib::labeltype, std::string> load_chunks_mapping(const std::string& filename) {
    std::unordered_map<hnswlib::labeltype, std::string> mapping;
    std::ifstream file(filename);
    std::string line;
    
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open chunks mapping file: " << filename << std::endl;
        return mapping;
    }
    
    while (std::getline(file, line)) {
        size_t colon_pos = line.find(':');
        if (colon_pos != std::string::npos) {
            try {
                hnswlib::labeltype id = std::stoul(line.substr(0, colon_pos));
                std::string chunk_text = line.substr(colon_pos + 1);
                mapping[id] = chunk_text;
            } catch (const std::exception& e) {
                std::cerr << "Error parsing line: " << line << " - " << e.what() << std::endl;
            }
        }
    }
    
    std::cout << "Loaded " << mapping.size() << " chunk mappings" << std::endl;
    return mapping;
}

int main() {
    // 加载chunks映射
    auto chunks_mapping = load_chunks_mapping("chunks_map.txt");
    
    // 创建空间度量（L2距离）
    hnswlib::L2Space space(768);
    
    // 加载索引
    hnswlib::HierarchicalNSW<float>* index;
    try {
        index = new hnswlib::HierarchicalNSW<float>(&space, "index.bin", false, 10000);
        std::cout << "Index loaded successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading index: " << e.what() << std::endl;
        return 1;
    }

    // 设置查询参数
    index->setEf(50); // 与Python代码中的p.set_ef(50)保持一致

    // 获取向量维度
    size_t dim = space.get_data_size() / sizeof(float);
    std::cout << "Vector dimension: " << dim << std::endl;

    // 交互式查询循环
    while (true) {
        std::cout << "\nEnter query text (or 'quit' to exit): " << std::endl;
        
        std::string query_text;
        std::getline(std::cin, query_text);
        
        if (query_text == "quit" || query_text == "exit") {
            break;
        }

        if (query_text.empty()) {
            continue;
        }

        try {
            // 获取文本的嵌入向量
            std::cout << "Generating embedding for: " << query_text << std::endl;
            std::vector<float> query_vector = get_embedding_from_text(query_text);
            
            // 检查向量维度
            if (query_vector.size() != dim) {
                std::cout << "Error: Expected " << dim << " dimensions, got " << query_vector.size() << std::endl;
                std::cout << "Generated vector: ";
                for (size_t i = 0; i < std::min(query_vector.size(), size_t(10)); ++i) {
                    std::cout << query_vector[i] << " ";
                }
                std::cout << "..." << std::endl;
                continue;
            }

            // 执行查询
            auto results = index->searchKnnCloserFirst(query_vector.data(), 3); // 查询最近的3个邻居
            
            std::cout << "Query results:" << std::endl;
            for(auto ld : results) {
                // std::pair<float, hnswlib::labeltype> top = results.top();
                float distance = ld.first;
                hnswlib::labeltype label = ld.second;
                // result.pop();
                std::cout << "Label: " << label << ", Distance: " << distance << std::endl;
                std::cout << "chunk: " << chunks_mapping[label] << std::endl;
                std::cout << "------------------------------------------------" << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Error during query: " << e.what() << std::endl;
        }
    }

    delete index;
    return 0;
}