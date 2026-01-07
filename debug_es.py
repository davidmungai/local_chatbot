from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")
print(f"Info: {es.info()}")

try:
    print("Checking existence of pdf-rag...")
    exists = es.indices.exists(index="pdf-rag")
    print(f"Exists: {exists}")
except Exception as e:
    print(f"Error checking existence: {e}")

try:
    print("Creating index with basic settings...")
    if not es.indices.exists(index="test-index"):
        es.indices.create(index="test-index")
    print("Created test-index")
except Exception as e:
    print(f"Error creating index: {e}")
