from dotenv import load_dotenv
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

# reopen the saved Chroma database from step 2
db = chromadb.PersistentClient(path="./darwin_chroma_db")
chroma_collection = db.get_or_create_collection("origin_of_species")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# same embedding model as ingestion
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# index over the existing vectors (nothing is re-embedded)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

# query engine: retrieve top 3 chunks, then the LLM writes the answer
llm = OpenAI()
query_engine = index.as_query_engine(
    llm=llm, response_mode="tree_summarize", similarity_top_k=3
)

response = query_engine.query("What does Darwin mean by natural selection?")
print(response)
print("Pages used:", [n.metadata.get("page_label") for n in response.source_nodes])
