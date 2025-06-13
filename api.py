from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# ✅ Allow all origins (for CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Debug: list visible files on Render
import os
@app.get("/")
def list_files():
    return {"files_in_server": os.listdir(".")}

# ✅ Input format
class Query(BaseModel):
    question: str
    image: Optional[str] = None  # base64 string (optional)

# ✅ Fallback POST endpoint
@app.post("/api/")
async def virtual_ta(query: Query):
    return {
        "answer": f"(Fallback) You asked: '{query.question}'. This is a sample response.",
        "links": [
            {
                "url": "https://discourse.onlinedegree.iitm.ac.in/",
                "text": "Visit Discourse for official answers"
            }
        ]
    }




  
  
