from urllib.parse import quote
import os
from glob import glob
from PIL import Image
from tqdm import tqdm
import chromadb
from transformers import ViTImageProcessor, ViTModel
from pathlib import Path
import torch

# 1. ChromaDB 연결
client = chromadb.PersistentClient(path=str(Path(__file__).resolve().parent / "chroma_data"))
collection_name = "foods"

# 기존 컬렉션 제거 후 새로 생성
try:
    client.delete_collection(collection_name)
except:
    pass

collection = client.create_collection(collection_name)

# 2. 이미지 임베딩 모델 로드
processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
model = ViTModel.from_pretrained('facebook/dino-vits16')
model.eval()

# 3. 이미지 리스트 불러오기
img_list = sorted(glob("test_images/*/*.jpg"))
print(f"🔍 총 {len(img_list)}장 이미지 로딩")

embeddings, metadatas, ids = [], [], []

# 4. 이미지 임베딩 및 ChromaDB 저장
for i, img_path in enumerate(tqdm(img_list)):
    try:
        img = Image.open(img_path).convert("RGB")
        cls = os.path.basename(os.path.dirname(img_path))

        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.pooler_output.squeeze().detach().numpy().tolist()

        # 상대 경로 URL-safe 인코딩
        relative_path = f"{cls}/{os.path.basename(img_path)}"
        encoded_path = quote(relative_path)

        embeddings.append(emb)
        metadatas.append({"uri": encoded_path, "name": cls})
        ids.append(str(i))

    except Exception as e:
        print(f"❌ 오류: {img_path}: {e}")

# 5. 컬렉션에 추가
collection.add(embeddings=embeddings, metadatas=metadatas, ids=ids)
print(f"✅ 완료: {len(ids)}건 추가됨")
