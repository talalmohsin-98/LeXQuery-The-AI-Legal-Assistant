import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# =========================
# CHROMA
# =========================

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vector_db = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=embeddings
)

# =========================
# LLM
# =========================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# =========================
# SYSTEM PROMPT
# =========================

system_prompt = """
You are LeXQuery, an expert AI legal assistant specialized in Pakistani law.
Rules:
- Answer primarily from the provided legal context
- Never invent laws, punishments, or section numbers
- Explain legal concepts in simple English
- Mention all relevant sections found in context
- If a section is directly asked, explain it clearly
- If partial information exists, still try to help carefully
- Only say 'I could not find this information in the provided documents.' if the
context truly lacks the answer
- Always mention section numbers accurately
- Be comprehensive and structured
"""

# =========================
# DETECT SECTION NUMBER
# =========================
def extract_section(question: str):

    patterns = [
        r'section\s+(\d+[A-Z]?)',
        r'u/s\s+(\d+[A-Z]?)',
        r'^(\d+[A-Z]?)$'
    ]

    question = question.lower()

    for pattern in patterns:

        match = re.search(pattern, question)
        if match:
            return match.group(1).upper()
        
        return None
    
# =========================
# RETRIEVAL
# =========================

def retrieve_documents(question: str):

    section = extract_section(question)
    docs = []

    # =========================
    # DIRECT SECTION SEARCH
    # =========================

    if section:
        print(f"Direct section lookup: {section}")
        docs = vector_db.similarity_search(
        query=question,
        k=10,
        filter={"section": section}
    )
    if docs:
        return docs
    
    # =========================
    # GENERAL LEGAL SEARCH
    # =========================

    docs = vector_db.similarity_search(
        question,
        k=15
    )

    return docs

# =========================
# FINAL ANSWER
# =========================

def get_answer(question: str):

    results = retrieve_documents(question)
    context = ""
    for r in results:
        context += f"""
            Section: {r.metadata.get('section')}
            Title: {r.metadata.get('title')}
            Source: {r.metadata.get('source')}
            Page: {r.metadata.get('page')}
            Content:
            {r.page_content}
            ---------------------
            """
        
    prompt = ChatPromptTemplate.from_messages([
    (
        "system", system_prompt
    ), 
    (
        "human",
        "Context:\n{context}\n\nQuestion: {question}"
    )
    ])

    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": question
    })
    return {
        "answer": response.content,
        "sources": [
        {
            "source": r.metadata.get("source"),
            "page": r.metadata.get("page"),
            "section": r.metadata.get("section"),
            "title": r.metadata.get("title")
        }
        for r in results
    ]
}