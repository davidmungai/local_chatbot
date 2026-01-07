"""
PDF → Chunking → Ollama Embeddings → Elasticsearch → RAG QA

Requirements:
  pip install langchain langchain-elasticsearch elasticsearch pypdf ollama

Prerequisites:
  - Elasticsearch running locally on http://localhost:9200
  - Ollama running locally on http://localhost:11434
  - Ollama models pulled:
        ollama pull nomic-embed-text
        ollama pull llama3
"""

from elasticsearch import Elasticsearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA


# -------------------------
# Configuration
# -------------------------

PDF_PATH = "data/NIPS-2017-attention-is-all-you-need-Paper.pdf"   # <-- change this
ES_URL = "http://localhost:9200"
INDEX_NAME = "pdf-rag"
OLLAMA_BASE_URL = "http://localhost:11434"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma3:latest"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# -------------------------
# 1. Load PDF
# -------------------------

print("Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print(f"Loaded {len(documents)} pages")


# -------------------------
# 2. Chunk PDF
# -------------------------

print("Chunking document...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")


# -------------------------
# 3. Elasticsearch Connection
# -------------------------

print("Connecting to Elasticsearch...")
es = Elasticsearch(ES_URL)


# -------------------------
# 4. Ollama Embeddings
# -------------------------

print("Initializing Ollama embeddings...")
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL
)


# -------------------------
# 5. Create / Load Vector Store
# -------------------------

print("Creating Elasticsearch vector store...")
vectorstore = ElasticsearchStore(
    client=es,
    index_name=INDEX_NAME,
    embedding=embeddings
)


# -------------------------
# 6. Ingest Chunks
# -------------------------

print("Indexing chunks (this may take a bit)...")
vectorstore.add_documents(chunks)
print("Indexing complete")


# -------------------------
# 7. Setup RAG Chain
# -------------------------

print("Setting up RAG pipeline...")
llm = Ollama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)


# -------------------------
# 8. Ask Questions
# -------------------------

while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    result = qa(query)

    print("\nAnswer:\n")
    print(result["result"])

    print("\nSources:\n")
    for doc in result["source_documents"]:
        page = doc.metadata.get("page", "N/A")
        source = doc.metadata.get("source", "unknown")
        print(f"- {source}, page {page}")