import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from pathlib import Path

load_dotenv()


# Same CHROMA_DIR path as ingest.py
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Connect to existing ChromaDB (no re-embedding)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
db = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=embeddings
)

system_prompt = """You are LeXQuery, an AI legal assistant 
specialized in Pakistani Law (PPC, CRPC, Constitution).

CONTEXT HANDLING:
- You will receive legal text from Pakistani law documents
- These documents use Urdu legal terms like qatl, diyat, 
  qisas, ta'zir, ikrah etc.
- Always explain these terms naturally in plain English
  within your answer

ANSWERING RULES:
- Answer ONLY from the provided context
- If the answer is not in the context, say exactly:
  'I could not find this information in the provided documents.'
- Write in clear, simple English for non-lawyers
- Explain legal terms when first used:
  example: qatl-e-amd (intentional murder)
- Be precise with section numbers and punishments
- Always begin with one simple sentence explaining 
   the concept in plain English before giving legal details
- Always mention the relevant section number when citing 
   a law or punishment. Format it like:
   'Under Section 302(a) PPC, the punishment is death as qisas'
- If multiple sections apply, mention each one separately
- Section numbers must come from the context provided,
  never from your own knowledge
- If multiple sections cover the same topic, 
   mention ALL of them from the context
- Do not stop at the first relevant section found
- Provide a comprehensive answer covering all 
   aspects present in the context

Never invent laws, sections, or punishments."""

def rewrite_query(question: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Pakistani legal search expert.
                    Rewrite the user question into an enhanced search query 
                    that includes relevant PPC/CRPC section numbers, Urdu legal 
                    terms, and related legal concepts.

                    IMPORTANT: Always respond in ENGLISH only.
                    Return ONLY the search query. No explanation."""),
        ("human", "Question: {question}")
    ])

    chain = rewrite_prompt | llm
    result = chain.invoke({"question": question})
    return result.content

def get_answer(question: str) -> dict:
    
    enhanced_query = rewrite_query(question)
    results = db.max_marginal_relevance_search(
        enhanced_query, 
        k=6,
        fetch_k=30
    )

    context = ""

    for r in results:
        filename = Path(r.metadata['source']).name
        context += r.page_content
        context += f"\nSource: {filename} | Page: {str(r.metadata['page'])}"
        context += "\n---\n"


    # Build prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    # Create LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Chain and invoke
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})

    # Return result
    return {
    "answer": response.content,
    "sources": [
            {
                "source": Path(r.metadata['source']).name,
                "page": r.metadata['page']
            }
            for r in results
        ]
    }

if __name__ == "__main__":
    questions = ["Can a woman file a case for harassment?"]
    enhanced = rewrite_query(questions[0])
    print(f"Enhanced query: {enhanced}")
    for question in questions:
        result = get_answer(question)
        print(f"\nQ: {question}")
        print(f"A: {result['answer']}")
        print("\nSources:")
        for source in result["sources"]:
            filename = Path(source['source']).name
            print(f"- {filename} | Page {source['page']}")