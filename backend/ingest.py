import re
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

# =========================
# PATHS
# =========================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

# =========================
# EMBEDDINGS
# =========================

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

# =========================
# CLEAN TEXT
# =========================

def clean_text(text: str) -> str:

    text = re.sub(r'\s+', ' ', text)
    text = text.replace("\x0c", " ")
    text = text.strip()

    return text

# =========================
# SECTION BASED SPLITTING
# =========================

def split_legal_sections(text: str):

    pattern = r'(\b\d+[A-Z]?\.\s+[A-Z][^\.\n]+)'

    matches = list(re.finditer(pattern, text))

    sections = []

    for i in range(len(matches)):

        start = matches[i].start()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        chunk = text[start:end].strip()

        sections.append(chunk)

    return sections

# =========================
# LOAD PDFS
# =========================

all_documents = []

for pdf in DATA_DIR.glob("*.pdf"):

    print(f"Loading {pdf.name}...")

    loader = PyPDFLoader(str(pdf))
    pages = loader.load()

    for page in pages:

        cleaned = clean_text(page.page_content)

        sections = split_legal_sections(cleaned)

        for section_text in sections:

            # Extract section number
            section_match = re.search(r'^(\d+[A-Z]?)\.', section_text)

            section_number = (
                section_match.group(1)
                if section_match else "Unknown"
            )

            # Extract title
            title_match = re.search(
                r'^\d+[A-Z]?\.\s+([^\.]+)',
                section_text
            )

            title = (
                title_match.group(1).strip()
                if title_match else "Unknown"
            )

            doc = Document(
                page_content=section_text,
                metadata={
                    "source": pdf.name,
                    "page": page.metadata.get("page", 0),
                    "section": section_number,
                    "title": title,
                    "act": pdf.stem.upper()
                }
            )

            all_documents.append(doc)

print(f"Total legal chunks created: {len(all_documents)}")

# =========================
# CREATE CHROMADB
# =========================

print("Legal ChromaDB created successfully!")