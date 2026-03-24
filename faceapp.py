import streamlit as st
import os
import sys

# 1. 라이브러리 로드 및 에러 핸들링
try:
    import cv2
    import mediapipe as mp
    import numpy as np
except ImportError as e:
    st.error(f"❌ 라이브러리 로드 실패: {e}")
    st.info("💡 해결 방법: requirements.txt에 'opencv-python-headless'가 있는지 확인하고, Python 버전을 3.10으로 설정했는지 체크하세요.")
    st.stop() # 에러 발생 시 이후 코드 실행 중단

# 2. 분석 엔진 로드 (FaceAnalyzer.py가 같은 폴더에 있어야 함)
try:
    from FaceAnalyzer import analyze_full_body
except ImportError:
    st.error("❌ 'FaceAnalyzer.py' 파일을 찾을 수 없습니다. GitHub에 함께 업로드했는지 확인해주세요.")
    st.stop()

# --- 여기서부터 웹 화면 구성 ---
st.set_page_config(page_title="AI 냉정 외모 분석기", page_icon="🤖")

st.title("🤖 AI 냉정 외모 분석기")
st.caption("당신의 중안부 비율과 어깨 너비를 객관적으로 분석합니다.")

uploaded_file = st.file_uploader("정면 사진을 업로드하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # 이미지 읽기
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # 분석용 임시 파일 저장
    temp_path = "temp_user_photo.jpg"
    cv2.imwrite(temp_path, image)
    
    with st.spinner("AI가 당신의 비율을 뜯어보는 중..."):
        result = analyze_full_body(temp_path)
        
    st.divider()
    
    if isinstance(result, dict):
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="분석 대상", use_container_width=True)
        
        with col2:
            st.write("### 📊 분석 결과")
            # 얼굴 분석 결과 출력
            f_res = result.get("얼굴", {})
            if f_res.get("상태") == "성공":
                mid = f_res.get("중안부", 0)
                st.metric("중안부 비율", f"{mid}")
                if mid < 0.6:
                    st.success("👶 **동안 등급: 만렙** (중안부가 매우 짧음)")
                else:
                    st.info("🧐 **성숙 등급: 보통**")
            
            # 전신 분석 결과 출력
            p_res = result.get("전신", {})
            if p_res.get("상태") == "성공":
                sh = p_res.get("어깨너비", 0)
                st.metric("어깨 너비 (px)", f"{sh}")
                st.write("💪 당당한 체형을 가지고 계시네요!")
    else:
        st.error(f"분석 중 오류 발생: {result}")

    # 임시 파일 삭제
    if os.path.exists(temp_path):
        os.remove(temp_path)