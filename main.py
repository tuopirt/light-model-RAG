

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


# processes file
def import_and_parse(filePath):
    elements = partition(filePath)
    page_texts = {}
    for el in elements:
        page_number = getattr(el.metadata, "page_number", None)
        page_number = int(page_number) if page_number else 1
        page_texts.setdefault(page_number, []).append(str(el))
    pages = []
    for page_num, texts in page_texts.items():
        combined_text = " ".join(texts).strip()
        pages.append({"text": combined_text, "page_number": page_num})
    return pages


# splits paragraph into chunks
def chunking(pages):
    enc = tiktoken.get_encoding("cl100k_base")
    chunks = []
    for page in pages:
        text = page["text"]
        tokens = enc.encode(text)
        for i in range(0, len(tokens), max_token_count):
            token_chunk = tokens[i:i + max_token_count]
            chunk_text = enc.decode(token_chunk)
            chunks.append({"chunk_text": chunk_text, "page_number": page["page_number"]})
    return chunks


# embeds
def embedding(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")  #can change this, current one is small & CPU-friendly but less efficient
    texts = [c["chunk_text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True)
    embedded_chunks = []
    for c, emb in zip(chunks, embeddings):
        embedded_chunks.append({
            "embedding": emb.tolist(),
            "chunk_text": c["chunk_text"],
            "page_number": c["page_number"]
        })
    return embedded_chunks


# stores the embedded chunks
def store_into_db(embeddedChunks):
    existing_ids = set(collection.get()["ids"] or [])
    new_count = 0
    for i, chunk in enumerate(embeddedChunks):
        uid = f"page{chunk['page_number']}_chunk{i}"
        if uid in existing_ids:
            continue
        collection.add(
            documents=[chunk["chunk_text"]],
            embeddings=[chunk["embedding"]],
            metadatas=[{"page_number": chunk["page_number"],
                        "n_tokens": len(chunk["chunk_text"].split())}],
            ids=[uid]
        )
        new_count += 1
    
    return collection



def main():
    # check to see if document already parsed for this case to save time
    if collection.count():
        print("Skipping PDF parsing and embedding")
    else:
        print("Begin parsing PDF...")
        pages = import_and_parse(pdfPath)
        print("Parsing complete")

        chunks = chunking(pages)
        print("Split into",len(chunks),"chunks")

        embedded_chunks = embedding(chunks)
        print("Chunks embedded")

        store_into_db(embedded_chunks)
        print("Stored in Chroma")
    # embed user question

    # loop interaction