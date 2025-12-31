import os
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_classic.chains import RetrievalQA

# Load environment variables
load_dotenv()

class RAGEngine:
    def __init__(self, data_dir="./data", persist_path="./vectorstore.json"):
        self.data_dir = data_dir
        self.persist_path = persist_path
        # gemma3 does not support embeddings, using nomic-embed-text
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vectorstore = None
        
        # Check if vectorstore exists
        if os.path.exists(self.persist_path):
            try:
                self.vectorstore = SKLearnVectorStore(
                    embedding=self.embeddings,
                    persist_path=self.persist_path,
                    serializer="json"
                )
            except Exception as e:
                print(f"Failed to load vectorstore: {e}")
        
    def ingest_data(self):
        """Loads data from the data directory and creates/updates the vector store."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            return "Data directory created. Please add files."

        documents = []
        # Load PDFs
        pdf_loader = DirectoryLoader(self.data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents.extend(pdf_loader.load())
        
        # Load Text files
        txt_loader = DirectoryLoader(self.data_dir, glob="**/*.txt", loader_cls=TextLoader)
        documents.extend(txt_loader.load())

        if not documents:
            return "No documents found in data directory."

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        self.vectorstore = SKLearnVectorStore.from_documents(
            documents=texts, 
            embedding=self.embeddings,
            persist_path=self.persist_path,
            serializer="json"
        )
        self.vectorstore.persist()
        return f"Ingested {len(documents)} documents and created vector store."

    def get_qa_chain(self):
        if not self.vectorstore:
            if os.path.exists(self.persist_path):
                 self.vectorstore = SKLearnVectorStore(
                        embedding=self.embeddings,
                        persist_path=self.persist_path,
                        serializer="json"
                    )
            else:
                return None
            
        llm = ChatOllama(model="gemma3", temperature=0)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        return qa_chain

    def query(self, question):
        qa_chain = self.get_qa_chain()
        if not qa_chain:
            return "Vector store not initialized. Please ingest data first."
        
        response = qa_chain.invoke({"query": question})
        return response
