

# Libraries
from unstructured.partition.auto import partition
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import chromadb
import numpy as np
import os
import tiktoken


# File paths abd folders, change directories here as needed
pdfPath = "./files/nvidia_quarter_report.pdf" # example file used as grounding
dbPath = os.path.abspath("./chroma_db") # using chroma db as our database
mdlPath = "model/mistral-7b-instruct-v0.2.Q3_K_M.gguf" # LLM used
threshold = 0.1 # hard-coded value change as needed
max_token_count = 300 # hard-coded token count


# initialize clients and storage directory
os.makedirs(dbPath, exsist_ok= True)

# db
chrom_client = chromadb.PersistentClient(path=dbPath)
collection = chrom_client.get_or_create_collection(
    name="rag_chunks",
    metadata={"hnsw:space": "cosine"}
)

# LLM
agent = Llama(model_path=mdlPath)


def import_and_parse(filePath):


def chunking(pages):


def embedding(chunks):


def store_into_db(embeddedChunks)




def main():
    # check to see if document already parsed for this case to save time

    # embed user question

    # loop interaction