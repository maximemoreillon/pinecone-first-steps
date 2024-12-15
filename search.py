# Import the Pinecone library
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from os import getenv
load_dotenv()

PINECONE_API_KEY = getenv("PINECONE_API_KEY")

# Initialize a Pinecone client with your API key
pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index("example-index")


# Define your query
query = "Tell me about the tech company known as Apple."

# Convert the query into a numerical vector that Pinecone can search with
query_embedding = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=[query],
    parameters={
        "input_type": "query"
    }
)

# Search the index for the three most similar vectors
results = index.query(
    namespace="example-namespace",
    vector=query_embedding[0].values,
    top_k=3,
    include_values=False,
    include_metadata=True
)

print(results)