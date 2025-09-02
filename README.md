# 向量数据库查询系统

一个结合 Python 和 C++ 的高效向量相似度搜索系统，使用 HNSW 算法进行近似最近邻搜索。
可以用于 Rag 应用的建库和搜索。

## 项目原理

本项目实现了一个完整的向量检索流水线：

离线建库部分：

1. **数据处理**：使用 LangChain 将文档（PDF/Markdown）分割成文本块
2. **向量化**：通过 Ollama 的 nomic-embed-text 模型生成 768 维文本嵌入向量
3. **索引构建**：使用 HNSW（Hierarchical Navigable Small World）算法构建高效索引

在线检索部分：

4. **快速查询**：C++ 程序加载索引并进行低延迟相似度搜索

## 环境要求

### Python 环境
```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install hnswlib numpy langchain pypdf2 ollama
```

### Ollama 设置
```bash
# 安装并启动 Ollama
安装ollama
ollama pull nomic-embed-text
nohup ollama serve &
```

### C++ 环境
- GCC 编译器（支持 C++17）

### HNSWLib C++ 库
```bash
# 在项目根目录下，下载 head-only HNSWLib C++ 库
git clone https://github.com/nmslib/hnswlib.git
```

## 文件说明

- `query.cpp`: C++ 查询程序源码
- `compile_query.sh`: 编译脚本
- `index.py`: Python 索引构建程序
- `embed.py`: 文本嵌入生成工具
- `chunks.py`: 文档分割处理模块
- `index.bin`: HNSW 索引文件（由 index.py 生成）
- `chunks_map.txt`: 文本块映射文件
- `data/`: 数据目录（放置 PDF/Markdown 文件）

## 完整运行步骤

### 1. 准备数据
将需要索引的文档（PDF 或 Markdown 文件）放入 `data/` 目录

### 2. 构建索引
```bash
python index.py
```

### 3. 编译 C++ 查询程序
```bash
chmod +x compile_query.sh
./compile_query.sh
```

### 4. 进行查询
```bash
./query_index
```

### 5. 查看结果
程序会返回最相似的 3 个文本块及其相似度距离（距离越小越相似）

## 技术特点

- **高效检索**：HNSW 算法提供近似最近邻搜索，支持大规模向量数据库
- **多格式支持**：支持 PDF 和 Markdown 文档格式
- **生产就绪**：C++ 查询程序确保低延迟和高性能
- **易于扩展**：模块化设计，可轻松替换嵌入模型或调整参数

## 后续拓展
1. 构建http线上服务提供rag服务
2. 在线服务增量更新
3. 分布式架构以及模块化拓展过滤、粗排、精排