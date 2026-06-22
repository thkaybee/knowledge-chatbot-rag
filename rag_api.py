from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

import requests

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS index
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Request model
class ChatRequest(BaseModel):
    question: str

# Health check
@app.get("/")
def home():
    return {
        "status": "RAG API Running"
    }

# Chat endpoint
@app.post("/chat")
def chat(request: ChatRequest):

    docs = db.similarity_search(
        request.question,
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are a helpful AI assistant.

Use ONLY the information provided in the context below.

Rules:
1. If the answer is present in the context, answer clearly.
2. If the answer is not present in the context, respond:
   "I could not find that information in the provided documents."
3. Do not make up information.
4. Do not use outside knowledge.

Context:
{context}

Question:
{request.question}

Answer:
"""

    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "default",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
    )

    answer = response.json()["choices"][0]["message"]["content"]

    return {
        "question": request.question,
        "answer": answer
    }
