import cv2
import numpy as np
import mediapipe as mp

def nothing(x):
    pass

cap = cv2.VideoCapture(1)
fgbg = cv2.createBackgroundSubtractorMOG2()

# ROI 영역 정의
roi_x = 52
roi_y = 0
roi_width = 500
roi_height = 310

# Mediapipe 손 감지 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=3, min_detection_confidence=0.3)

# 트랙바를 위한 윈도우 생성
cv2.namedWindow('Frame')
cv2.createTrackbar('Threshold', 'Frame', 0, 20000, nothing)  # 초기값 0, 최대값 20000

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 트랙바에서 현재 threshold 값을 가져옴
    threshold = cv2.getTrackbarPos('Threshold', 'Frame')

    # 배경 차분을 사용하여 전경 마스크 생성
    fgmask = fgbg.apply(frame)

    # 특정 영역에 대한 침입 감지
    roi = fgmask[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    intrusion_detected = np.sum(roi) > threshold

    # Mediapipe를 사용하여 손 감지
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                # 손가락 랜드마크 좌표 계산
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                
                # 손가락 랜드마크가 ROI 영역 내에 있는지 확인
                if roi_x < x < roi_x + roi_width and roi_y < y < roi_y + roi_height and intrusion_detected:
                    print("Intrusion detected!")
                    cv2.putText(frame, "Warning: Intrusion detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    break

            # 손 랜드마크 그리기
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 원본 프레임에 ROI 경계를 그립니다.
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (255, 0, 0), 2)

    # 감지된 마커가 있는 이미지를 표시합니다.
    cv2.imshow('Frame', frame)
    cv2.imshow('Foreground Mask', fgmask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
