# backend/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests
import torch
import chromadb
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path

app = FastAPI()

# 절대 경로에서 정적 이미지 서빙 경로 마운트
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/images", StaticFiles(directory=os.path.join(BASE_DIR, "data")), name="images")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class ImageQuery(BaseModel):
    image_url: str

processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
model = ViTModel.from_pretrained('facebook/dino-vits16')
model.eval()

client = chromadb.PersistentClient(path=str(Path(__file__).resolve().parent / "chroma_data"))
collection = client.get_or_create_collection("foods")

@app.post("/query")
def query_image(query: ImageQuery):
    print("🔍 요청 받은 URL:", query.image_url)

    if not query.image_url or not query.image_url.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid image_url")

    try:
        response = requests.get(query.image_url, stream=True, timeout=5)
        response.raise_for_status()
        img = Image.open(response.raw).convert("RGB")
    except Exception as e:
        print(f"❌ 이미지 로딩 실패: {e}")
        raise HTTPException(status_code=400, detail=f"Image loading failed: {e}")

    try:
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs)
        embedding = output.pooler_output.squeeze().tolist()
    except Exception as e:
        print(f"❌ 임베딩 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    try:
        result = collection.query(query_embeddings=[embedding], n_results=4)
    except Exception as e:
        print(f"❌ ChromaDB 질의 실패: {e}")
        raise HTTPException(status_code=500, detail=f"ChromaDB query failed: {e}")

    return [
        {**meta, "distance": result["distances"][0][i]}
        for i, meta in enumerate(result["metadatas"][0])
    ]

@app.get("/list")
def list_items(limit: int = 20):
    try:
        results = collection.get(include=["ids", "metadatas"], limit=limit)
        return results["metadatas"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Listing failed: {e}")