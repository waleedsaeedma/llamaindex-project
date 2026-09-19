from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(input_dir="./data")
documents = reader.load_data()

print(f"Loaded {len(documents)} document(s)")
print(documents[0].metadata)
print(documents[0].text[:300])
