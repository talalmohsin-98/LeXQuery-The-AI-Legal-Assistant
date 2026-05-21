# ⚖️ LeXQuery — AI Legal Assistant for Pakistani Law

A production-ready RAG (Retrieval-Augmented Generation) system 
that answers legal questions about Pakistani law with source citations.
Built with LangChain, ChromaDB, FastAPI, and OpenAI.

---

## 🎯 What It Does

- Ask any question about Pakistani law in plain English
- Searches through PPC, CRPC, and Constitution automatically  
- Returns accurate answers with section numbers cited
- Shows exact source document and page number for every answer
- Never makes up laws — only answers from real documents

---

## 🏗️ Architecture

- User Question (plain English)
↓
- Query Rewriting (GPT-4o-mini enriches with legal terms)
↓
- ChromaDB Vector Search (finds relevant legal chunks)
↓
- GPT-4o reads chunks and answers in plain English
↓
- Answer + Source Citations returned

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | OpenAI GPT-4o |
| Embeddings | OpenAI text-embedding-3-large |
| Vector Database | ChromaDB |
| RAG Framework | LangChain |
| Backend API | FastAPI |
| PDF Processing | PyPDF |
| Frontend | HTML, CSS, JavaScript |

---

## 📁 Project Structure

LeXQuery/
├── backend/
│   ├── ingest.py      # PDF loading, chunking, ChromaDB ingestion
│   ├── rag.py         # RAG chain, query rewriting, answer generation
│   └── main.py        # FastAPI endpoints
├── data/
│   ├── ppc.pdf        # Pakistan Penal Code
│   ├── crpc.pdf       # Code of Criminal Procedure
│   └── constitution.pdf
├── frontend/
│   └── index.html     # Chat interface
├── .env               # API keys (not committed)
└── requirements.txt

---

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/talalmohsin-98/LeXQuery-The-AI-Legal-Assistant.git
cd LeXQuery-The-AI-Legal-Assistant
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**

Create a `.env` file in the root folder:
OPENAI_API_KEY=your_openai_api_key_here

**5. Ingest documents**
```bash
cd backend
python ingest.py
```

**6. Start the API**
```bash
uvicorn main:app --reload
```

**7. Open the frontend**

Open `frontend/index.html` in your browser.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat` | Ask a legal question |
| GET | `/health` | Check API status |
| GET | `/docs` | Swagger UI documentation |

**Example request:**
```json
POST /chat
{
  "question": "What is the punishment for murder in Pakistan?"
}
```

**Example response:**
```json
{
  "answer": "Under Section 302(a) PPC, the punishment for 
             qatl-e-amd (intentional murder) is death as qisas...",
  "sources": [
    {"source": "ppc.pdf", "page": 105},
    {"source": "ppc.pdf", "page": 110}
  ]
}
```

---

## 💡 Key Features

- **Query Rewriting**: User's plain English is automatically 
  enriched with Pakistani legal terminology before searching
- **MMR Search**: Maximal Marginal Relevance ensures diverse 
  chunk retrieval instead of repetitive results
- **Ditto Filter**: Removes low-quality table chunks from 
  legal schedule pages
- **Citation System**: Every answer includes source document 
  and page number
- **Input Validation**: Empty questions rejected with 422 error

---

## 👤 Author

**Muhammad Talal Mohsin**  
Applied AI/ML Developer  
[GitHub](https://github.com/talalmohsin-98)
[LinkedIn]https://www.linkedin.com/in/talal-mohsin-kaleem/
