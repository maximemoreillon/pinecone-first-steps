from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from os import getenv
import time
from openai import OpenAI

load_dotenv()

INDEX_NAME="openai-experiments"

PINECONE_API_KEY = getenv("PINECONE_API_KEY")


MODEL = "text-embedding-3-small"


client = OpenAI()  
pc = Pinecone(api_key=PINECONE_API_KEY)

input = [
        "A cardigan is a type of knitted garment that has an open front, and is worn like a jacket.",
        "A sweater (North American English) or pullover, also called a jersey or jumper is a piece of clothing, typically with long sleeves, made of knitted or crocheted material that covers the upper part of the body. When sleeveless, the garment is often called a slipover, tank top, or sweater vest.",
        "A T-shirt (also spelled tee shirt, or tee for short) is a style of fabric shirt named after the T shape of its body and sleeves. Traditionally, it has short sleeves and a round neckline, known as a crew neck, which lacks a collar. T-shirts are generally made of stretchy, light, and inexpensive fabric and are easy to clean. The T-shirt evolved from undergarments used in the 19th century and, in the mid-20th century, transitioned from undergarments to general-use casual clothing.",
        "A dress shirt, button shirt, button-front, button-front shirt, or button-up shirt is a garment with a collar and a full-length opening at the front, which is fastened using buttons or shirt studs. A button-down or button-down shirt is a dress shirt with a button-down collar – a collar having the ends fastened to the shirt with buttons.",
        "A coat is typically an outer garment for the upper body, worn by any gender for warmth or fashion. Coats typically have long sleeves and are open down the front, and closing by means of buttons, zippers, hook-and-loop fasteners (AKA velcro), toggles, a belt, or a combination of some of these. Other possible features include collars, shoulder straps, and hoods",
        "Shorts are a garment worn over the pelvic area, circling the waist and splitting to cover the upper part of the legs, sometimes extending down to the knees but not covering the entire length of the leg. They are called shorts because they are a shortened version of trousers, which cover the entire leg, but not the foot. Shorts are typically worn in warm weather or in an environment where comfort and airflow are more important than the protection of the legs.",
        "Trousers (British English), slacks, or pants (American, Canadian and Australian English) are an item of clothing worn from the waist to anywhere between the knees and the ankles, covering both legs separately (rather than with cloth extending across both legs as in robes, skirts, dresses and kilts)",
    ]



res = client.embeddings.create(
    input=input, model=MODEL
)

embeds = [record.embedding for record in res.data]

ids = [str(n) for n in range(len(input))]
meta = [{'text': line} for line in input]
to_upsert = zip(ids, embeds, meta)







# check if index already exists (it shouldn't if this is your first run)
if INDEX_NAME not in pc.list_indexes().names():
    # if does not exist, create index
    spec = ServerlessSpec(cloud="aws", region="us-east-1")
    pc.create_index(
        INDEX_NAME,
        dimension=len(embeds[0]),  # dimensionality of text-embed-3-small
        metric='dotproduct',
        spec=spec
    )
    # wait for index to be initialized
    while not pc.describe_index(INDEX_NAME).status['ready']:
        time.sleep(1)

# connect to index
index = pc.Index(INDEX_NAME)
time.sleep(1)


index.upsert(vectors=list(to_upsert))