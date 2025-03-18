from pinecone import Pinecone
import os

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))  # Or replace with your API key directly
index_name = os.environ.get("PINECONE_INDEX_NAME")

# Connect to the index
index = pc.Index(index_name)

# Delete all vectors
index.delete(delete_all=True)
print("All vectors deleted successfully!")
