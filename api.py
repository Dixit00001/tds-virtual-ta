from fastapi import FastAPI, UploadFile, File, Request
from pydantic import BaseModel
from typing import Optional, List
import base64
import openai
import faiss
import json
import numpy as np
import pytesseract
from PIL import Image
from io import BytesIO

# === Load OpenAI Key ===
openai.api_key = "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"  # 🔁 Replace with your key

# === Load Chunks and Index ===
with open("chunks.jsonl", "r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]

index = faiss.read_index("index.faiss")

# === Embedding Parameters ===
EMBED_MODEL = "text-embedding-3-small"
TOP_K = 5

# === FastAPI Setup ===
app = FastAPI()

class Query(BaseModel):
    question: str
    image: Optional[str] = None

@app.post("/api/")
async def answer_question(query: Query):
    question_text = query.question

    # === Step 1: Extract text from base64 image if provided ===
    if query.image:
        try:
            image_data = base64.b64decode(query.image)
            image = Image.open(BytesIO(image_data))
            extracted = pytesseract.image_to_string(image)
            question_text += " " + extracted
        except Exception as e:
            return {"error": f"Image OCR failed: {str(e)}"}

    # === Step 2: Get embeddings ===
    embed_resp = openai.embeddings.create(
        model=EMBED_MODEL,
        input=question_text
    )
    query_vec = np.array(embed_resp.data[0].embedding, dtype="float32").reshape(1, -1)

    # === Step 3: Search similar chunks ===
    D, I = index.search(query_vec, TOP_K)
    context_chunks = [chunks[i] for i in I[0]]
    context_text = "\n---\n".join([c["text"] for c in context_chunks])

    # === Step 4: RAG prompt ===
    prompt = f"""You are a helpful TA for the Tools in Data Science course.
Use the context below to answer the student's question.
Context:
{context_text}

Question: {question_text}

Answer:"""

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "user", "content": prompt}],
    )

    final_answer = response.choices[0].message.content.strip()

    # === Step 5: Link metadata ===
    links = [{"url": f"https://fake.discourse/iitm/{c['id']}", "text": f"From {c['source']}"} for c in context_chunks]

    return {
        "answer": final_answer,
        "links": links
    }
