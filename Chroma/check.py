# 확인용 스크립트
from chromadb import PersistentClient
from pathlib import Path

client = PersistentClient(path=str(Path("backend/chroma_data")))  # 경로 일치!
collection = client.get_or_create_collection("foods")
data = collection.get(include=["metadatas"], limit=5)
print(data)
