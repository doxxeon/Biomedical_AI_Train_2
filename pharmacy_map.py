import streamlit as st
import pandas as pd
import pymongo
import pydeck as pdk

st.set_page_config(layout="wide")
st.title("💊 전국 약국 위치 지도")

# ✅ MongoDB 연결
client = pymongo.MongoClient("mongodb://localhost:27017")
db = client["pharmacy_db"]
collection = db["pharmacy_info"]

# ✅ MongoDB → DataFrame
data = pd.DataFrame(list(collection.find({
    "x": {"$ne": ""}, "y": {"$ne": ""}
})))

# ✅ 컬럼명 정리
data = data.rename(columns={
    "place_name": "약국명",
    "road_address_name": "주소",
    "phone": "전화번호",
    "x": "경도",
    "y": "위도"
})

# ✅ NaN 제거 & 타입 변환
data = data.dropna(subset=["위도", "경도"])
data["위도"] = data["위도"].astype(float)
data["경도"] = data["경도"].astype(float)

# ✅ 📍 지역 리스트 뽑기 (주소 앞 부분)
data["지역"] = data["주소"].str.extract(r"(서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주)")
region_list = sorted(data["지역"].dropna().unique())

# ✅ 🎛️ 지역 필터 UI
selected_region = st.selectbox("📍 지역을 선택하세요", ["전체 보기"] + region_list)

# ✅ 🔍 약국명 텍스트 검색
search_keyword = st.text_input("🔎 약국명을 검색하세요").strip()

# ✅ 🔍 선택한 지역만 필터링
if selected_region != "전체 보기":
    filtered = data[data["지역"] == selected_region]
else:
    filtered = data

if search_keyword:
    filtered = filtered[filtered["약국명"].str.contains(search_keyword, case=False, na=False)]

# ⛑️ 위도/경도 float 변환 (문자열 제거 포함)
filtered["위도"] = pd.to_numeric(filtered["위도"], errors="coerce")
filtered["경도"] = pd.to_numeric(filtered["경도"], errors="coerce")

# 💥 NaN 제거
filtered = filtered.dropna(subset=["위도", "경도"])

# ✅ 지도용 데이터 정리
filtered_map_data = filtered.rename(columns={
    "위도": "latitude",
    "경도": "longitude",
    "약국명": "name"
})[["latitude", "longitude", "name"]].dropna()

# ✅ NaN 제거
filtered_map_data = filtered_map_data.dropna(subset=["latitude", "longitude"])

# ✅ 숫자형만 남기기
filtered_map_data = filtered_map_data[
    (filtered_map_data["latitude"].apply(lambda x: isinstance(x, (int, float)))) &
    (filtered_map_data["longitude"].apply(lambda x: isinstance(x, (int, float))))
]

filtered_map_data = filtered_map_data.reset_index(drop=True)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=filtered_map_data,
    get_position='[longitude, latitude]',
    get_color='[255, 0, 0, 160]',
    get_radius=100,
    pickable=True,
)

# ✅ 지도 중심: 검색 시 확대, 전체 보기일 땐 기본 줌
if len(filtered_map_data) > 0:
    view_state = pdk.ViewState(
        latitude=filtered_map_data["latitude"].mean(),
        longitude=filtered_map_data["longitude"].mean(),
        zoom=11 if search_keyword else 9,
        pitch=0
    )
else:
    view_state = pdk.ViewState(latitude=36.5, longitude=127.5, zoom=6)  # 기본 중심값

if len(filtered_map_data) == 0:
    st.warning("❌ 검색 결과가 없습니다.")
else:
    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "{name}"}
    ))

# ✅ 상세 표
with st.expander("📋 약국 상세 리스트 보기"):
    st.dataframe(filtered[["약국명", "주소", "전화번호"]])