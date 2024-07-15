import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector

# Initialize webcam
cap = cv2.VideoCapture(0)
# Initialize HandDetector
detector = HandDetector(maxHands=5, detectionCon=0.8)  # 최대 5개의 손을 인식

# 임계값 설정 (이 값을 조정하여 손이 너무 가까워졌을 때를 결정)
AREA_THRESHOLD = 100000  # 예시 값, 실제 상황에 맞게 조정 필요

# 손 ID 저장
tracked_hand_bbox = None

while True:
    success, img = cap.read()
    if not success:
        break

    # Find hands
    hands, img = detector.findHands(img)

    if hands:
        # 인식된 손들을 면적 기준으로 정렬
        hands.sort(key=lambda x: x['bbox'][2] * x['bbox'][3], reverse=True)
        print('hand:', hands)

        # 손마다 바운딩 박스를 표시
        for hand in hands:
            x, y, w, h = hand['bbox']
            cvzone.putTextRect(img, f'Area: {w * h}', (x, y - 10), scale=1, thickness=2, colorR=(255, 0, 0))
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if tracked_hand_bbox is None:
            # 면적이 가장 큰 손의 바운딩 박스를 저장
            tracked_hand_bbox = hands[0]['bbox']
        
        # 추적 중인 손을 찾기
        hand_to_track = None
        for hand in hands:
            if hand['bbox'] == tracked_hand_bbox:
                hand_to_track = hand
                break
        
        if hand_to_track:
            # Get the bounding box of the tracked hand
            x, y, w, h = hand_to_track['bbox']
            # Calculate the area of the bounding box
            area = w * h

            # 면적이 임계값을 초과하면 경고 메시지 표시
            if area > AREA_THRESHOLD:
                cvzone.putTextRect(img, "Oops! Too close, trying to steal the ice cream?", (50, 50), scale=1, thickness=2, colorR=(0, 0, 255))
                # 여기서 로봇을 제어하는 코드를 추가할 수 있습니다.
                # 예: 로봇을 n 차 뒤로 이동시키는 코드
                # move_robot_backward(n)

            # Display the area on the image
            cvzone.putTextRect(img, f'Tracked Area: {int(area)}', (50, 100), scale=2, thickness=2, colorR=(255, 0, 0))
            
            # 추적 중인 손을 강조 표시
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4)  # 빨간색 두꺼운 사각형
            cvzone.putTextRect(img, 'Tracking', (x, y - 50), scale=2, thickness=2, colorR=(0, 0, 255))
            
            
        else:
            # 추적 중인 손을 찾을 수 없는 경우, 초기화
            tracked_hand_bbox = None
    else:
        # 손이 없으면 ID 초기화
        tracked_hand_bbox = None

    # Display the image
    cv2.imshow("Image", img)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()

# import cv2
# import mediapipe as mp
# import numpy as np

# # MediaPipe Hands 초기화
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(max_num_hands=5, min_detection_confidence=0.8)
# mp_draw = mp.solutions.drawing_utils

# # 웹캠 초기화
# cap = cv2.VideoCapture(1)

# # 손의 고유 ID를 추적하기 위한 변수
# tracked_hand_id = None
# AREA_THRESHOLD = 0.1  # 예시 값, 실제 상황에 맞게 조정 필요 (단위: 비율)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # 프레임을 RGB로 변환
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)

#     if results.multi_hand_landmarks:
#         hand_ids = []
#         for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
#             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#             # 랜드마크를 배열로 변환
#             landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])

#             # 손의 랜드마크를 기준으로 고유한 ID를 생성
#             hand_id = hash(landmarks.tostring())
#             hand_ids.append(hand_id)

#             # 손의 바운딩 박스 계산
#             h, w, c = frame.shape
#             cx_min, cy_min = int(min(landmarks[:, 0]) * w), int(min(landmarks[:, 1]) * h)
#             cx_max, cy_max = int(max(landmarks[:, 0]) * w), int(max(landmarks[:, 1]) * h)
#             area = (cx_max - cx_min) * (cy_max - cy_min) / (w * h)  # 면적 비율 계산

#             if tracked_hand_id is None:
#                 tracked_hand_id = hand_id

#             if hand_id == tracked_hand_id:
#                 # 추적 중인 손 강조 표시
#                 cv2.rectangle(frame, (cx_min, cy_min), (cx_max, cy_max), (0, 0, 255), 4)  # 빨간색 두꺼운 사각형
#                 cv2.putText(frame, 'Tracking', (cx_min, cy_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#                 cv2.putText(frame, f'ID: {hand_id}', (cx_min, cy_min - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#                 # 면적이 임계값을 초과하면 경고 메시지 표시
#                 if area > AREA_THRESHOLD:
#                     cv2.putText(frame, "Oops! Too close, trying to steal the ice cream?", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#                     # 여기서 로봇을 제어하는 코드를 추가할 수 있습니다.
#                     # 예: 로봇을 n 차 뒤로 이동시키는 코드
#                     # move_robot_backward(n)
#             else:
#                 # 다른 손 표시
#                 cv2.rectangle(frame, (cx_min, cy_min), (cx_max, cx_max), (0, 255, 0), 2)
#                 cv2.putText(frame, f'ID: {hand_id}', (cx_min, cy_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
#     # Display the image
#     cv2.imshow("Hand Tracking", frame)

#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
