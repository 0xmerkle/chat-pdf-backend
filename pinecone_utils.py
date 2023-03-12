from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import uuid

load_dotenv()

import os

open_ai_key = os.environ.get("OPENAI_API_KEY")
pinecone_key = os.environ.get("PINECONE_API_KEY")
index_name = os.environ.get("INDEX_NAME")


def get_embeddings(all_text):
    # Initialize the Langchain client
    openai = OpenAIEmbeddings(openai_api_key=open_ai_key)

    vecs = openai.embed_documents(texts=all_text, chunk_size=1000)

    return vecs


def load_pages_to_pinecone(pages) -> None:
    print("pages", pages)
    # Set up the Pinecone client
    pinecone.init(api_key=pinecone_key, environment="us-west1-gcp")

    # Create a new index if it doesn't exist
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=1636, metric="euclidean")
    embeddings = OpenAIEmbeddings(openai_api_key=open_ai_key)
    docsearch = Pinecone.from_documents(pages, embeddings, index_name=index_name)
    print(docsearch)
    return index_name
