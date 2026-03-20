import cv2
import mediapipe as mp
import numpy as np
import os

# 1. Mediapipe 모듈 초기화 (안전한 로드 방식)
try:
    # 얼굴 분석용 (FaceMesh)
    mp_face_mesh = mp.solutions.face_mesh
    
    # 전신 분석용 (Pose)
    mp_pose = mp.solutions.pose
except Exception as e:
    print(f"라이브러리 로드 오류: {e}")
    mp_face_mesh = None
    mp_pose = None

def calculate_dist(p1, p2):
    """두 지점 사이의 거리 계산"""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def analyze_full_body(image_path):
    if mp_face_mesh is None or mp_pose is None:
        return "에러: Mediapipe가 정상적으로 설치되지 않았습니다."
    
    if not os.path.exists(image_path):
        return f"에러: '{image_path}' 사진 파일을 찾을 수 없습니다."

    # 분석 결과 데이터 구조
    final_data = {
        "얼굴": {"상태": "대기", "중안부": 0, "미간": 0},
        "전신": {"상태": "대기", "어깨너비": 0}
    }

    # 이미지 읽기 및 RGB 변환
    raw_image = cv2.imread(image_path)
    if raw_image is None:
        return "에러: 이미지를 읽을 수 없습니다."
    
    # 이미지 해상도 정보 (픽셀 좌표 변환용)
    ih, iw, _ = raw_image.shape
    image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    # --- PART 1: 얼굴 분석 (FaceMesh) ---
    try:
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True
        ) as face_mesh:
            
            results_face = face_mesh.process(image_rgb)
            
            if results_face.multi_face_landmarks:
                landmarks = results_face.multi_face_landmarks[0].landmark
                
                def get_f_coords(idx):
                    return [landmarks[idx].x * iw, landmarks[idx].y * ih]

                # 핵심 비율 계산 (기존 로직 동일)
                mid_top = get_f_coords(168) # 미간
                nose_tip = get_f_coords(1)   # 코끝
                chin_bottom = get_f_coords(152) # 턱끝
                eye_l = get_f_coords(133)   # 왼쪽눈 안쪽
                eye_r = get_f_coords(362)   # 오른쪽눈 안쪽
                eye_l_out = get_f_coords(33) # 왼쪽눈 바깥

                mid_ratio = calculate_dist(mid_top, nose_tip) / calculate_dist(nose_tip, chin_bottom)
                eye_ratio = calculate_ratio(calculate_dist(eye_l, eye_r), calculate_dist(eye_l_out, eye_l))

                final_data["얼굴"] = {
                    "상태": "성공",
                    "중안부": round(mid_ratio, 2),
                    "미간": round(eye_ratio, 2)
                }
            else:
                final_data["얼굴"]["상태"] = "인식 실패 (정면 사진 필요)"
    except Exception as e:
        final_data["얼굴"]["상태"] = f"에러: {e}"

    # --- PART 2: 전신 분석 (Pose) ---
    try:
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2, # 정확도 우선 (0, 1, 2)
            min_detection_confidence=0.5
        ) as pose:
            
            results_pose = pose.process(image_rgb)
            
            if results_pose.pose_landmarks:
                p_landmarks = results_pose.pose_landmarks.landmark
                
                # 11번: 왼쪽 어깨, 12번: 오른쪽 어깨 (정면 기준)
                shoulder_l = [p_landmarks[11].x * iw, p_landmarks[11].y * ih]
                shoulder_r = [p_landmarks[12].x * iw, p_landmarks[12].y * ih]
                
                # 어깨 너비 계산 (픽셀 거리)
                shoulder_width = calculate_dist(shoulder_l, shoulder_r)
                
                final_data["전신"] = {
                    "상태": "성공",
                    "어깨너비": round(shoulder_width, 1) # 픽셀 단위
                }
            else:
                final_data["전신"]["상태"] = "인식 실패 (발끝까지 나오게 찍으세요)"
    except Exception as e:
        final_data["전신"]["상태"] = f"에러: {e}"

    return final_data

def calculate_ratio(dist1, dist2):
    return dist1 / dist2 if dist2 != 0 else 0

# --- 메인 실행부 ---
if __name__ == "__main__":
    # 전신과 얼굴이 모두 잘 나온 사진을 사용하세요.
    target_photo = 'face.jpg' 
    
    print("\n" + "="*50)
    print(" [ AI 종합 외모 분석 시스템 가동 ] ")
    print(" (얼굴 비율 + 전신 체형 데이터를 추출합니다) ")
    print("="*50)
    
    result = analyze_full_body(target_photo)
    
    import json
    # 결과 데이터를 보기 좋게 출력
    print(json.dumps(result, indent=4, ensure_ascii=False))
    
    print("="*50 + "\n")