from pinecone import Pinecone
import aioboto3
from uuid import uuid4
from tenacity import retry, wait_exponential, stop_after_attempt
from asyncio import Semaphore
import asyncio
import json

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

# Global semaphore for rate limiting
upsert_semaphore = Semaphore(5)

# Initialize clients
index = None  # Pinecone index
bedrock_session = None

def initialize_pinecone():
    """Initialize Pinecone index with serverless configuration"""
    global index
    try:
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=1024,
                metric="cosine",
                spec={"serverless": serverless_config}
            )
            print(f"Created new serverless index: {index_name}")
        
        index = pc.Index(index_name)  # Synchronous instance
    except Exception as e:
        print(f"Pinecone initialization failed: {str(e)}")
        raise

async def initialize_clients():
    """Initialize all async clients"""
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
    
    try:
        async with upsert_semaphore:
            chunk_id = str(uuid4())
            
            async with bedrock_session.client(
                service_name='bedrock-runtime',
                region_name='us-east-1'
            ) as bedrock:
                input_data = {
                    "inputText": chunk,
                    "dimensions": 1024,
                    "normalize": True
                }
                response = await bedrock.invoke_model(
                    modelId=modelId,
                    contentType="application/json",
                    accept="*/*",
                    body=json.dumps(input_data)
                )
                response_body = await response['body'].read()
                response_json = json.loads(response_body)
                embedding = response_json['embedding']

            # Run Pinecone upsert in a separate thread
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, index.upsert, [{
                "id": chunk_id,
                "values": embedding,
                "metadata": {"chunk": chunk}
            }])

            print(f"[Async Updated] {chunk[:50]}...")
            return True
            
    except Exception as e:
        print(f"UPSERT ERROR: {str(e)}")
        raise  

async def startup():
    await initialize_clients()

try:
    asyncio.run(startup())
except Exception as e:
    print(f"Application failed to start: {str(e)}")
    exit(1)
