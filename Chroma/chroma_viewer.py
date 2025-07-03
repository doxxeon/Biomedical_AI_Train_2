import streamlit as st
import chromadb
from pathlib import Path
import pandas as pd

# ChromaDB 클라이언트 연결
db_path = Path(__file__).resolve().parent / "backend" / "chroma_data"
client = chromadb.PersistentClient(path=str(db_path))
collection = client.get_or_create_collection("foods")

# 메타데이터 불러오기
try:
    results = collection.get(include=["metadatas"])
    metadatas = results.get("metadatas", [])
except Exception as e:
    st.error(f"❌ ChromaDB 불러오기 오류: {e}")
    metadatas = []

# Streamlit 페이지 설정
st.set_page_config(page_title="📦 ChromaDB 데이터 테이블")
st.title("📦 ChromaDB Collection Viewer")
st.markdown("ChromaDB에 저장된 이미지 메타데이터를 아래 표로 확인하세요.")

if not metadatas:
    st.warning("📭 No data found in ChromaDB.")
else:
    # Pandas DataFrame으로 출력
    df = pd.DataFrame(metadatas)
    st.dataframe(df, use_container_width=True)
