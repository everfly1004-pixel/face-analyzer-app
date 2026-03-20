import streamlit as st
import cv2
import numpy as np
from FaceAnalyzer import analyze_full_body # 기존 코드 파일명

# 페이지 설정
st.set_page_config(page_title="AI 냉정 외모 분석기", page_icon="🤖")

st.title("🤖 AI 냉정 외모 분석기")
st.subheader("당신의 황금 비율과 체형을 낱낱이 파헤쳐 드립니다.")

uploaded_file = st.file_uploader("정면 전신 사진을 업로드하세요 (JPG, PNG)", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # 1. 이미지 로드
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # 임시 저장
    temp_path = "temp_user_photo.jpg"
    cv2.imwrite(temp_path, image)
    
    with st.spinner("AI가 정밀 분석 중입니다..."):
        # 2. 분석 엔진 가동 (기존 코드 호출)
        result = analyze_full_body(temp_path)
        
    # 3. 결과 출력
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="분석된 사진", use_container_width=True)
        
    with col2:
        st.write("### 📊 분석 리포트")
        if result["얼굴"]["상태"] == "성공":
            mid = result["얼굴"]["중안부"]
            st.metric("중안부 비율", f"{mid}")
            # 팩폭 멘트 로직
            if mid < 0.6:
                st.info("👶 **[동안 등급: 만렙]** \n중안부가 엄청 짧네요! 평생 동안 소리 듣고 살 팔자입니다.")
            elif mid > 1.1:
                st.warning("🧐 **[성숙 등급: 고인물]** \n중안부가 길어 신뢰감을 주는 상입니다. 노안 소리 주의!")
        
        if result["전신"]["상태"] == "성공":
            sh = result["전신"]["어깨너비"]
            st.metric("어깨 너비 수치", f"{sh}px")
            st.success("💪 **[체형 분석]** 어깨가 아주 당당하시군요!")

    # 4. 공유 유도 (이미지 생성 기능은 Pillow 라이브러리 추가 필요)
    st.button("📲 결과 카카오톡으로 공유하기 (준비 중)")
    st.button("📸 인스타 스토리용 이미지 저장 (준비 중)")