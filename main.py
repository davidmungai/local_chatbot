from elasticsearch import Elasticsearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA


PDF_PATH = "data/NIPS-2017-attention-is-all-you-need-Paper.pdf"   # <-- change this
ES_URL = "http://localhost:9200"
INDEX_NAME = "pdf-rag"
OLLAMA_BASE_URL = "http://localhost:11434"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma3"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print(f"Loaded {len(documents)} pages")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_documents(documents)
es = Elasticsearch(ES_URL)

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL
)
vectorstore = ElasticsearchStore(
    client=es,
    index_name=INDEX_NAME,
    embedding=embeddings
)



vectorstore.add_documents(chunks)

llm = OllamaLLM(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

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