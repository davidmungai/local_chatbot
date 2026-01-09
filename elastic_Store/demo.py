import time
import sys
import os

# Add the current directory to sys.path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from elasticsearch import Elasticsearch
from secure_search import secure_retrieve, get_ollama_embedding, INDEX_NAME, ES_URL

def index_sample_data():
    """
    Indexes sample documents with security metadata (tenant_id, allowed_roles).
    """
    es = Elasticsearch(ES_URL)
    
    # 1. Define Mapping (Optional but good practice for dense_vector)
    if not es.indices.exists(index=INDEX_NAME):
        print(f"Creating index {INDEX_NAME}...")
        es.indices.create(index=INDEX_NAME, body={
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "tenant_id": {"type": "keyword"},
                    "allowed_roles": {"type": "keyword"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 768, # nomic-embed-text dimension
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        })
    else:
        print(f"Index {INDEX_NAME} exists. Adding sample data...")

    # 2. Sample Documents
    documents = [
        {
            "id": "doc1",
            "text": "The project 'Alpha' confidential financial results show a 20% increase in revenue. (Finance Department Only)",
            "tenant_id": "tenant-A",
            "allowed_roles": ["finance", "admin"],
            "popularity": 10
        },
        {
            "id": "doc2",
            "text": "The company cafeteria menu for next week includes tacos and pizza. (Public to Tenant A)",
            "tenant_id": "tenant-A",
            "allowed_roles": ["employee", "finance", "admin"],
            "popularity": 5
        },
        {
            "id": "doc3",
            "text": "Tenant B's secret strategy for 2026. (Tenant B Only)",
            "tenant_id": "tenant-B",
            "allowed_roles": ["admin"],
            "popularity": 8
        }
    ]

    # 3. Embed and Index
    for doc in documents:
        print(f"Embedding and indexing doc: {doc['id']}")
        embedding = get_ollama_embedding(doc['text'])
        doc['embedding'] = embedding
        
        es.index(index=INDEX_NAME, id=doc['id'], document=doc)
        
    print("Refresh index...")
    es.indices.refresh(index=INDEX_NAME)
    print("Data indexed successfully.\n")

def run_demo():
    # Setup data
    try:
        index_sample_data()
    except Exception as e:
        print(f"Failed to index headers (is Elasticsearch running?): {e}")
        return

    query = "What are the financial results?"
    print(f"QUERY: '{query}'")
    print("-" * 50)

    # Scenario 1: Authorized User (Tenant A, Finance)
    user_finance = {
        "userId": "user1",
        "tenantId": "tenant-A",
        "roles": ["finance"]
    }
    print(f"\nScenario 1: User {user_finance['userId']} (Roles: {user_finance['roles']}, Tenant: {user_finance['tenantId']})")
    result = secure_retrieve(query, user_finance)
    print("RESULT:")
    print(result if result else "[No documents returned]")

    # Scenario 2: Unauthorized User (Tenant A, Employee - No Finance access)
    user_employee = {
        "userId": "user2",
        "tenantId": "tenant-A",
        "roles": ["employee"]
    }
    print(f"\nScenario 2: User {user_employee['userId']} (Roles: {user_employee['roles']}, Tenant: {user_employee['tenantId']})")
    result = secure_retrieve(query, user_employee)
    print("RESULT:")
    print(result if result else "[No documents returned (Correct - Access Denied)]")

    # Scenario 3: Wrong Tenant (Tenant B Admin)
    user_tenant_b = {
        "userId": "user3",
        "tenantId": "tenant-B",
        "roles": ["admin"]
    }
    print(f"\nScenario 3: User {user_tenant_b['userId']} (Roles: {user_tenant_b['roles']}, Tenant: {user_tenant_b['tenantId']})")
    result = secure_retrieve(query, user_tenant_b)
    print("RESULT:")
    print(result if result else "[No documents returned (Correct - Wrong Tenant)]")

if __name__ == "__main__":
    run_demo()
