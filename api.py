import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import base64
import io
import json
import pytesseract
from PIL import Image

app = FastAPI()

# ✅ Enable CORS for frontend or Render testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (OK for demo)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Debug route to confirm files are present on Render
@app.get("/")
def home():
    return {"files_in_server": os.listdir(".")}

# ✅ Define POST input format
class Query(BaseModel):
    question: str
    image: Optional[str] = None  # base64 string (optional)

# ✅ Main logic
@app.post("/api/")
async def get_answer(query: Query):
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        # Load your chunks data
        with open("chunks.jsonl", "r", encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f]

        texts = [chunk["text"] for chunk in chunks]
        sources = [chunk.get("source", "") for chunk in chunks]
        ids = [chunk.get("id", "") for chunk in chunks]

        model = SentenceTransformer("all-MiniLM-L6-v2")
        index = faiss.read_index("index.faiss")

        question = query.question

        # Optional image-based OCR
        if query.image:
            image_data = base64.b64decode(query.image)
            image = Image.open(io.BytesIO(image_data))
            extracted_text = pytesseract.image_to_string(image)
            question += " " + extracted_text.strip()

        # Search relevant chunks
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
        return {
            "error": "Something went wrong. Possibly missing files or invalid model setup.",
            "details": str(e)
        }



  
  
