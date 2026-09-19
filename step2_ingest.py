import chromadb
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# load the book (same as step 1)
documents = SimpleDirectoryReader(input_dir="./data").load_data()

# Chroma: a persistent vector store saved on disk
db = chromadb.PersistentClient(path="./darwin_chroma_db")
chroma_collection = db.get_or_create_collection("origin_of_species")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# ingestion pipeline: split -> embed -> store
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_overlap=0),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ],
    vector_store=vector_store,
)

nodes = pipeline.run(documents=documents)

print(f"Created {len(nodes)} nodes")
print(f"Chroma collection now holds {chroma_collection.count()} items")
print(nodes[0].metadata.get("page_label"))
print(nodes[0].text[:200])
