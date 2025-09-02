# embed.py
import ollama
import sys

EMBEDDING_MODEL = 'nomic-embed-text'

def embed(text: str) -> list[float]:
    result = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=text
    )
    embedding = result.model_dump()["embedding"]
    # print(len(embedding))
    return embedding

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # 获取命令行参数（合并所有参数为一个查询字符串）
        query = ' '.join(sys.argv[1:])
        embedding = embed(query)
        # 输出为逗号分隔的格式，便于C++解析
        print(','.join(map(str, embedding)))
    else:
        print("Usage: python embed.py <query_text>")