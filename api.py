import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import base64
import io
import json
from PIL import Image
import pytesseract

app = FastAPI()

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"files": os.listdir(".")}

class Query(BaseModel):
    question: str
    image: Optional[str] = None

@app.post("/api/")
async def get_answer(query: Query):
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        # Try to load chunks
        with open("chunks.jsonl", "r", encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f]

        texts = [chunk["text"] for chunk in chunks]
        sources = [chunk.get("source", "") for chunk in chunks]

        model = SentenceTransformer("all-MiniLM-L6-v2")
        index = faiss.read_index("index.faiss")

        question = query.question

        # If image is given, extract text
        if query.image:
            image_data = base64.b64decode(query.image)
            image = Image.open(io.BytesIO(image_data))
            extracted_text = pytesseract.image_to_string(image)
            question += " " + extracted_text.strip()

        embedding = model.encode([question]).astype("float32")
        _, I = index.search(embedding, 3)

        selected = [texts[i] for i in I[0]]
        selected_sources = [sources[i] for i in I[0]]

        return {
            "answer": selected[0],
            "links": [
                {
                    "url": f"https://discourse.onlinedegree.iitm.ac.in/{src}",
                    "text": selected[i]
                }
                for i, src in enumerate(selected_sources)
            ]
        }

    except Exception as e:
        # If anything fails, fallback gracefully
        return {
            "answer": "(Fallback) I'm having trouble loading the data right now.",
            "links": [
                {
                    "url": "https://discourse.onlinedegree.iitm.ac.in/",
                    "text": "Visit Discourse for official answers"
                }
            ],
            "error": str(e)
        }
