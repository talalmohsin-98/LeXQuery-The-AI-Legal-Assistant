# importing Required Modules

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path

# loading .env file
load_dotenv()

# =========================
# PATHS
# =========================

BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

# =========================
# EMBEDDING MODEL
# =========================

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

# =========================
# CHECK IF CHROMA DB EXISTS
# =========================

if CHROMA_DIR.exists():

    print("Loading Existing ChromaDB...")

    # Load existing DB from disk
    db = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )

else:

    print("Creating ChromaDB for first time...")

    # =========================
    # LOAD PDF FILES
    # =========================

    all_docs = []

    for pdf in DATA_DIR.glob("*.pdf"):

        loader = PyPDFLoader(str(pdf))

        print(f"{pdf.name} Loaded!")

        docs = loader.load()

        all_docs.extend(docs)

    # =========================
    # TEXT SPLITTING
    # =========================

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    final_docs = text_splitter.split_documents(all_docs)

    print(f"Total pages loaded: {len(all_docs)}")
    print(f"Total chunks created: {len(final_docs)}")

    # =========================
    # CREATE CHROMA DB
    # =========================

    db = Chroma.from_documents(
        documents=final_docs,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    print("ChromaDB Created and Saved!")

# =========================
# SIMILARITY SEARCH
# =========================

query = "What is the punishment for murder in Pakistan?"

results = db.similarity_search(query, k=3)

# =========================
# PRINT RESULTS
# =========================

print("\nSimilarity Search Results:\n")

for i, result in enumerate(results, start=1):

    print(f"\nResult {i}")
    print("-" * 50)

    print(result.page_content[:1000])

    print("\nMetadata:")
    print(result.metadata)