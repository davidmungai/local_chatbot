import os
import requests
import json
from typing import Dict, Any, List
from elasticsearch import Elasticsearch

# Configuration
# In a production environment, these should be loaded from environment variables
ES_URL = os.getenv("ES_URL", "http://localhost:9200")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = "nomic-embed-text"
INDEX_NAME = "pdf-rag"

# Initialize Elasticsearch client
# Ensure usage of CA certs or authentication if moving to production HTTPS
es_client = Elasticsearch(ES_URL)

def get_ollama_embedding(text: str) -> List[float]:
    """
    Generates an embedding for the provided text using the Ollama API.
    Endpoint: POST /api/embeddings
    """
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    payload = {
        "model": EMBEDDING_MODEL,
        "prompt": text
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["embedding"]
    except requests.exceptions.RequestException as e:
        # Log error in production
        print(f"Error communicating with Ollama: {e}")
        raise

def secure_retrieve(query_text: str, user_context: Dict[str, Any]) -> str:
    """
    Executes a secure vector retrieval query against Elasticsearch.
    
    This function enforces the security boundary at the database level (Elasticsearch).
    It guarantees that no documents are retrieved or sent to the LLM context unless
    they explicitly match the user's tenant_id and roles.
    
    Args:
        query_text: The user's input/question.
        user_context: A dictionary containing:
            - userId: str
            - tenantId: str
            - roles: List[str]
            
    Returns:
        A single string containing the concatenated text of authorized matching documents.
    """
    
    # 1. Validate User Context
    tenant_id = user_context.get("tenantId")
    user_roles = user_context.get("roles", [])
    
    if not tenant_id:
        raise ValueError("Security Violation: Missing tenantId in user context.")
    
    if not user_roles:
        # If user has no roles, they might see nothing, or public docs if allowed.
        # Assuming restrictive by default:
        print("Warning: User has no roles assigned.")

    # 2. Generate Query Embedding
    query_vector = get_ollama_embedding(query_text)
    
    # 3. Construct Secure Elasticsearch Query (kNN)
    # The 'filter' clause inside 'knn' ensures that only accessible documents are considered 
    # as candidates for the approximate kNN search. This is crucial for performance and security.
    es_query = {
        "knn": {
            "field": "embedding",
            "query_vector": query_vector,
            "k": 5,                 # Return top 5 matches
            "num_candidates": 50,    # Search 50 candidates per shard for recall
            "filter": [
                # Security Boundary: Tenant Isolation
                {
                    "term": {
                        "tenant_id": tenant_id
                    }
                },
                # Security Boundary: Role-Based Access Control
                # Matches if the document's 'allowed_roles' contains ANY of the user's roles.
                {
                    "terms": {
                        "allowed_roles": user_roles
                    }
                }
            ]
        },
        # Optimization: Only fetch the necessary fields
        "_source": ["text", "id", "metadata"] 
    }
    
    # 4. Execute Search
    try:
        response = es_client.search(index=INDEX_NAME, body=es_query)
    except Exception as e:
        print(f"Elasticsearch query failed: {e}")
        return "An error occurred while retrieving information."

    # 5. Process Results into Context String
    hits = response.get("hits", {}).get("hits", [])
    
    if not hits:
        # DO NOT hallucinate context. 
        # If ES returned nothing (due to filters or no correlation), return empty context.
        return ""
        
    context_parts = []
    for hit in hits:
        source_doc = hit.get("_source", {})
        text = source_doc.get("text", "")
        # Add basic separation/metadata if useful for the LLM
        context_parts.append(text)
        
    # Join all authorized document chunks
    llm_context = "\n\n".join(context_parts)
    
    return llm_context
