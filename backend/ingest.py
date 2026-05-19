from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"

all_docs = []

for pdf in DATA_DIR.glob("*.pdf"):
    loader = PyPDFLoader(pdf)
    print(f"{pdf} Loaded")
    docs = loader.load()
    all_docs.extend(docs)

text = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
"""
Chunk Size: 
    Complete document is divided into mentioned chunk size for storage and retreival. So that when referenced, instead of calling the whole document again and again, that specifc chunk is called.

Chunk Overlap:
    The number of overlapped words so that no context is missed. Basically this creates an intersectioin among chunks which is very important for legal matters.

"""

final_doc = text.split_documents(all_docs)
print(f"Total pages loaded: {len(all_docs)}")
print(f"Total chunks created: {len(final_doc)}")
print(final_doc[0].page_content)
print(final_doc[0].metadata)