from pinecone import Pinecone, Index
import aioboto3
from uuid import uuid4
from tenacity import retry, wait_exponential, stop_after_attempt
from asyncio import Semaphore
import asyncio
import json
import concurrent.futures

# Initialize Pinecone client
pc = Pinecone(
    api_key="pcsk_6BwgYz_ADdw4fhbmhdCMF28chYaTrH64hKXeVn4y7xYHPCALaq6KhGsvkaaevDcQQDSPbK"
)

# Configuration
modelId = "amazon.titan-embed-text-v2:0"
index_name = "qucoon-realtimerag"

# Serverless configuration
serverless_config = {
    "cloud": "aws",
    "region": "us-west-1"
}



# Initialize clients - These are synchronous, so we don't need async here.
index: Index = None  # Type hint for clarity
bedrock_session = None

def initialize_pinecone():
    """Initialize Pinecone index with serverless configuration"""
    global index
    try:
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=1536,  # Corrected dimension for Titan Embeddings V2
                metric="cosine",
                spec={"serverless": serverless_config}
            )
            print(f"Created new serverless index: {index_name}")

        index = pc.Index(index_name)  # Synchronous instance
        print(f"Connected to index: {index_name}")
    except Exception as e:
        print(f"Pinecone initialization failed: {str(e)}")
        raise

def initialize_clients():
    """Initialize  clients"""
    global bedrock_session
    try:
        initialize_pinecone()  # Keep this synchronous
        bedrock_session = aioboto3.Session()
        print("Clients initialized successfully")
    except Exception as e:
        print(f"Client initialization failed: {str(e)}")
        raise

@retry(wait=wait_exponential(multiplier=1, min=2, max=10),
       stop=stop_after_attempt(3),
       reraise=True)
async def async_update_db(chunk: str):
    """Asynchronously process and upsert a chunk with retries"""
    global index, bedrock_session

    # Create the semaphore *inside* the async function
    if not hasattr(async_update_db, 'upsert_semaphore'):
        async_update_db.upsert_semaphore = Semaphore(5)


    try:
        async with async_update_db.upsert_semaphore:
            chunk_id = str(uuid4())

            async with bedrock_session.client(
                service_name='bedrock-runtime',
                region_name='us-east-1'
            ) as bedrock:
                body = json.dumps({"inputText": chunk})
                response = await bedrock.invoke_model(
                    modelId=modelId,
                    contentType="application/json",
                    accept="application/json",
                    body=body
                )

                response_body_bytes = await response.get('body').read() # Await the read() method!
                response_body = json.loads(response_body_bytes) # Now pass bytes to json.loads
                embedding = response_body.get('embedding')


            # Correct upsert call
            to_upsert = [(chunk_id, embedding, {"chunk": chunk})]  # Correct format
            index.upsert(vectors=to_upsert) # Use the synchronous upsert


            print(f"[Async Updated] {chunk[:50]}...")
            return True

    except Exception as e:
        print(f"UPSERT ERROR: {str(e)}")
        raise

# No need for asyncio.run at the module level. We will call startup from sound.py.
async def startup():
   initialize_clients()