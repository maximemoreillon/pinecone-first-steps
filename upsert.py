# Import the Pinecone library
from dotenv import load_dotenv
from pinecone import Pinecone
from config import INDEX_NAME
from os import getenv
load_dotenv()

PINECONE_API_KEY = getenv("PINECONE_API_KEY")

# Initialize a Pinecone client with your API key
pc = Pinecone(api_key=PINECONE_API_KEY)


data = [
    {"id": "vec1", "text": "A cardigan is a type of knitted garment that has an open front, and is worn like a jacket."},
    {"id": "vec2", "text": "A sweater (North American English) or pullover, also called a jersey or jumper is a piece of clothing, typically with long sleeves, made of knitted or crocheted material that covers the upper part of the body. When sleeveless, the garment is often called a slipover, tank top, or sweater vest."},
    {"id": "vec3", "text": "A T-shirt (also spelled tee shirt, or tee for short) is a style of fabric shirt named after the T shape of its body and sleeves. Traditionally, it has short sleeves and a round neckline, known as a crew neck, which lacks a collar. T-shirts are generally made of stretchy, light, and inexpensive fabric and are easy to clean. The T-shirt evolved from undergarments used in the 19th century and, in the mid-20th century, transitioned from undergarments to general-use casual clothing."},
    {"id": "vec4", "text": "A dress shirt, button shirt, button-front, button-front shirt, or button-up shirt is a garment with a collar and a full-length opening at the front, which is fastened using buttons or shirt studs. A button-down or button-down shirt is a dress shirt with a button-down collar – a collar having the ends fastened to the shirt with buttons."},
    {"id": "vec5", "text": "A coat is typically an outer garment for the upper body, worn by any gender for warmth or fashion. Coats typically have long sleeves and are open down the front, and closing by means of buttons, zippers, hook-and-loop fasteners (AKA velcro), toggles, a belt, or a combination of some of these. Other possible features include collars, shoulder straps, and hoods"},
    {"id": "vec6", "text": "Shorts are a garment worn over the pelvic area, circling the waist and splitting to cover the upper part of the legs, sometimes extending down to the knees but not covering the entire length of the leg. They are called shorts because they are a shortened version of trousers, which cover the entire leg, but not the foot. Shorts are typically worn in warm weather or in an environment where comfort and airflow are more important than the protection of the legs."},
    {"id": "vec7", "text": "Trousers (British English), slacks, or pants (American, Canadian and Australian English) are an item of clothing worn from the waist to anywhere between the knees and the ankles, covering both legs separately (rather than with cloth extending across both legs as in robes, skirts, dresses and kilts)."}
]


# Embed your data
embeddings = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=[d['text'] for d in data],
    parameters={
        "input_type": "passage",
        "truncate": "END"
    }
)

index = pc.Index(INDEX_NAME)


# Prepare the records for upsert
# Each contains an 'id', the embedding 'values', and the original text as 'metadata'
records = []
for d, e in zip(data, embeddings):
    records.append({
        "id": d['id'],
        "values": e['values'],
        "metadata": {'text': d['text']}
    })

# Upsert the records into the index
index.upsert(
    vectors=records,
    namespace="example-namespace"
)