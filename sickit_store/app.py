import streamlit as st
import os
from rag_engine import RAGEngine

st.set_page_config(page_title="RAG Chatbot (Ollama)", page_icon="🤖")

st.title("🤖 RAG Chatbot (Ollama: gemma3)")

# Sidebar for configuration and data ingestion
with st.sidebar:
    st.header("Data Ingestion")
    st.write("Place your .pdf or .txt files in the `data` folder.")
    
    if st.button("Ingest Data"):
        with st.spinner("Ingesting data..."):
            engine = RAGEngine()
            result = engine.ingest_data()
            st.success(result)

# Main chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        engine = RAGEngine()
        response = engine.query(prompt)
        
        if isinstance(response, dict) and "result" in response:
            answer = response["result"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            with st.expander("Source Documents"):
                for doc in response["source_documents"]:
                    st.write(f"Source: {doc.metadata.get('source', 'Unknown')}")
                    st.write(doc.page_content)
        else:
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
