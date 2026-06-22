from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests

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

while True:
    question = input("\nQuestion (type 'exit' to quit): ")

    if question.lower() == "exit":
        break

    # Retrieve top 3 relevant chunks
    docs = db.similarity_search(question, k=3)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Improved RAG Prompt
    prompt = f"""
You are a helpful AI assistant.

Use ONLY the information provided in the context below to answer the question.

Rules:
1. If the answer is present in the context, answer clearly and concisely.
2. If the answer is not present in the context, respond exactly with:
   "I could not find that information in the provided documents."
3. Do not make up information.
4. Do not use outside knowledge.
5. Summarize information when appropriate.

Context:
{context}

Question:
{question}

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

    print("\n===== ANSWER =====")
    print(answer)

    print("\n===== RETRIEVED CONTEXT =====")
    for i, doc in enumerate(docs, 1):
        print(f"\nChunk {i}:")
        print(doc.page_content[:300])
