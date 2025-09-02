import hnswlib
import numpy as np
from chunks import get_chunks
from embed import embed

dim = 768
max_num_elements = 10000

# Get chunks from data folder
print("Getting chunks from data folder...")
chunks = get_chunks("data")
num_elements = len(chunks)
print(f"Found {num_elements} chunks")

# Generate embeddings for each chunk
print("Generating embeddings...")
embeddings = []
for i, chunk in enumerate(chunks):
    if i % 10 == 0:
        print(f"Embedding chunk {i+1}/{num_elements}")
    embedding_vector = embed(chunk)
    embeddings.append(embedding_vector)

print(f"Embedding chunk done {num_elements}/{num_elements}")

# Convert to numpy array
data = np.float32(embeddings)
ids = np.arange(num_elements)

# Declaring index
p = hnswlib.Index(space = 'l2', dim = dim) # possible options are l2, cosine or ip

# Initializing index - the maximum number of elements should be known beforehand
p.init_index(max_elements = max_num_elements, ef_construction = 200, M = 16, allow_replace_deleted = True)

# Element insertion (can be called several times):
p.add_items(data, ids, -1, True)

# Controlling the recall by setting ef:
p.set_ef(50) # ef should always be > k

p.save_index("index.bin")

print("Index saved successfully")

# Also save a simple text mapping for C++
with open("chunks_map.txt", "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        # Replace newlines with spaces for easier parsing
        chunk_line = chunk.replace("\n", " ").replace("\r", " ")
        f.write(f"{i}:{chunk_line}\n")

print("Ids and chunks mapping saved successfully")
