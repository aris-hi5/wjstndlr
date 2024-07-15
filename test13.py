import cv2
import numpy as np
from ultralytics import YOLO
import torch

# YOLOv8 모델 로드
model = YOLO('new_custom_m_freeze8.pt')  # YOLOv8 모델을 자동으로 다운로드합니다

# 모델을 GPU로 이동
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    model.to(device)

# GPU 사용 여부 출력
print(f'Using device: {device}')

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(0)  # 0은 기본 카메라를 의미합니다.

# 트랙바 창 생성
cv2.namedWindow('Trackbars')

# 트랙바 생성
cv2.createTrackbar('Threshold1', 'Trackbars', 50, 255, lambda x: None)
cv2.createTrackbar('Threshold2', 'Trackbars', 150, 255, lambda x: None)
cv2.createTrackbar('Diff_Thresh', 'Trackbars', 45, 255, lambda x: None)
cv2.createTrackbar('Confidence', 'Trackbars', 50, 100, lambda x: None)
cv2.createTrackbar('Hue Min', 'Trackbars', 0, 179, lambda x: None)
cv2.createTrackbar('Hue Max', 'Trackbars', 179, 179, lambda x: None)
cv2.createTrackbar('Sat Min', 'Trackbars', 0, 255, lambda x: None)
cv2.createTrackbar('Sat Max', 'Trackbars', 255, 255, lambda x: None)
cv2.createTrackbar('Val Min', 'Trackbars', 0, 255, lambda x: None)
cv2.createTrackbar('Val Max', 'Trackbars', 255, 255, lambda x: None)

# 영역 정의
roi_x_large = 52
roi_y_large = 0
roi_width_large = 500
roi_height_large = 310

roi_x_medium = 270
roi_y_medium = 0
roi_width_medium = 270
roi_height_medium = 60

roi_x_small = 464
roi_y_small = 118
roi_width_small = 35
roi_height_small = 35

# 초기 배경 이미지 변수
initial_gray = None
post_cleanup_gray = None
detection_enabled = False
cleanup_detection_enabled = False

while True:
    # 프레임 캡처
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera.")
        break

    # 현재 프레임을 그레이스케일로 변환 및 블러 적용
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)

    # HSV 변환
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 트랙바에서 현재 임계값 읽기
    diff_thresh = cv2.getTrackbarPos('Diff_Thresh', 'Trackbars')
    threshold1 = cv2.getTrackbarPos('Threshold1', 'Trackbars')
    threshold2 = cv2.getTrackbarPos('Threshold2', 'Trackbars')
    confidence_threshold = cv2.getTrackbarPos('Confidence', 'Trackbars') / 100.0
    h_min = cv2.getTrackbarPos('Hue Min', 'Trackbars')
    h_max = cv2.getTrackbarPos('Hue Max', 'Trackbars')
    s_min = cv2.getTrackbarPos('Sat Min', 'Trackbars')
    s_max = cv2.getTrackbarPos('Sat Max', 'Trackbars')
    v_min = cv2.getTrackbarPos('Val Min', 'Trackbars')
    v_max = cv2.getTrackbarPos('Val Max', 'Trackbars')
    
    # 초기값 설정 또는 해제
    if cv2.waitKey(1) & 0xFF == ord('s'):
        
           # 초기 배경 설정
            initial_gray = frame_gray
            detection_enabled = True
            print("Initial background set, detection enabled.")
    
    # 검출 활성화된 경우에만 실행
    if detection_enabled and initial_gray is not None and not cleanup_detection_enabled:
        # 초기 이미지와 현재 이미지의 차이 계산
        frame_delta = cv2.absdiff(initial_gray, frame_gray)
        
        # 차이 이미지의 임계값 적용
        _, diff_mask = cv2.threshold(frame_delta, diff_thresh, 255, cv2.THRESH_BINARY)
        diff_mask = cv2.dilate(diff_mask, None, iterations=2)

        # Canny 엣지 검출
        edges = cv2.Canny(frame_gray, threshold1, threshold2)
        edges = cv2.dilate(edges, None, iterations=1)
        edges = cv2.bitwise_and(edges, edges, mask=diff_mask)

        # HSV 색상 범위에 따른 마스크 생성
        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])
        hsv_mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

        # 탐지 영역 마스크 생성 (마커와 별 인식 영역 제외)
        detection_mask = np.zeros(diff_mask.shape, dtype=np.uint8)
        detection_mask[roi_y_large:roi_y_large + roi_height_large, roi_x_large:roi_x_large + roi_width_large] = 255
        detection_mask[roi_y_medium:roi_y_medium + roi_height_medium, roi_x_medium:roi_x_medium + roi_width_medium] = 0
        detection_mask[roi_y_small:roi_y_small + roi_height_small, roi_x_small:roi_x_small + roi_width_small] = 0

        # 최종 마스크 적용
        combined_mask = cv2.bitwise_or(diff_mask, edges)
        combined_mask = cv2.bitwise_and(combined_mask, combined_mask, mask=detection_mask)
        combined_mask = cv2.bitwise_and(combined_mask, hsv_mask)

        # 윤곽선 검출
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_objects = []

        for cnt in contours:
            if cv2.contourArea(cnt) > 80:  # 최소 면적 기준
                x, y, w, h = cv2.boundingRect(cnt)
                # 탐지된 객체가 지정된 탐지 영역 내에 있는지 확인
                if (roi_x_large <= x <= roi_x_large + roi_width_large and
                    roi_y_large <= y <= roi_y_large + roi_height_large):
                    detected_objects.append((x, y, w, h))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, 'Trash Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # YOLOv8 객체 검출 수행
        results = model(frame)

        # ROI에 바운딩 박스 그리기
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls)
                label = model.names[cls_id]
                confidence = box.conf.item()  # 신뢰도 추출

                # 신뢰도가 트랙바에서 설정한 값보다 높은 객체만 표시
                if confidence >= confidence_threshold:
                    bbox = box.xyxy[0].tolist()  # 바운딩 박스를 리스트로 변환
                    x1, y1, x2, y2 = map(int, bbox)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # 텍스트 위치 계산
                    text = f'{label} {confidence:.2f}'
                    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                    # 텍스트가 바운딩 박스 내부, 오른쪽 상단에 표시되도록 위치 조정
                    text_x = x2 - text_width if x2 - text_width > 0 else x1
                    text_y = y1 - 2 if y1 - 2 > text_height else y1 + text_height + 2

                    # 텍스트 배경 상자 그리기
                    cv2.rectangle(frame, (text_x, text_y - text_height - baseline), (text_x + text_width, text_y + baseline), (0, 255, 0), cv2.FILLED)

                    # 텍스트 그리기
                    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                    # 투명한 컵이 인식된 경우 쓰레기로 표시
                    if label == 'cup' or label == 'star':
                        cv2.putText(frame, 'Trash Detected', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        detected_objects.append((x1, y1, x2 - x1, y2 - y1))

        # 쓰레기가 계속 존재하는지 표시
        if detected_objects:
            cv2.putText(frame, 'Trash Present', (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # 원본 프레임과 결합된 마스크를 나란히 보여줌
        combined_frame = np.hstack((frame, cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)))

        # 프레임을 화면에 표시
        cv2.imshow('Original and Combined Mask', combined_frame)

    # 'q' 키를 누르면 루프를 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()





# import cv2
# import numpy as np

# # 비디오 캡처 객체 생성
# cap = cv2.VideoCapture(1)  # 0은 기본 카메라를 의미합니다.

# # 트랙바 창 생성
# cv2.namedWindow('Trackbars')

# # 트랙바 생성
# cv2.createTrackbar('Threshold1', 'Trackbars', 50, 255, lambda x: None)
# cv2.createTrackbar('Threshold2', 'Trackbars', 150, 255, lambda x: None)
# cv2.createTrackbar('Diff_Thresh', 'Trackbars', 55, 255, lambda x: None)
# cv2.createTrackbar('Hue Min', 'Trackbars', 0, 179, lambda x: None)
# cv2.createTrackbar('Hue Max', 'Trackbars', 179, 179, lambda x: None)
# cv2.createTrackbar('Sat Min', 'Trackbars', 0, 255, lambda x: None)
# cv2.createTrackbar('Sat Max', 'Trackbars', 255, 255, lambda x: None)
# cv2.createTrackbar('Val Min', 'Trackbars', 0, 255, lambda x: None)
# cv2.createTrackbar('Val Max', 'Trackbars', 255, 255, lambda x: None)

# # 영역 정의
# roi_x_large = 60
# roi_y_large = 0
# roi_width_large = 470
# roi_height_large = 310

# roi_x_medium = 270
# roi_y_medium = 0
# roi_width_medium = 270
# roi_height_medium = 60

# roi_x_small = 464
# roi_y_small = 118
# roi_width_small = 35
# roi_height_small = 35

# # 초기 배경 이미지 변수ㅂ
# initial_gray = None
# post_cleanup_gray = None
# detection_enabled = False
# cleanup_detection_enabled = False

# while True:
#     # 프레임을 읽기
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to capture frame from camera.")
#         break
#     cv2.imshow('frame',frame)
#     # 현재 프레임을 그레이스케일로 변환 및 블러 적용
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)

#     # HSV 변환
#     hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # 트랙바에서 현재 임계값 읽기
#     diff_thresh = cv2.getTrackbarPos('Diff_Thresh', 'Trackbars')
#     threshold1 = cv2.getTrackbarPos('Threshold1', 'Trackbars')
#     threshold2 = cv2.getTrackbarPos('Threshold2', 'Trackbars')
#     h_min = cv2.getTrackbarPos('Hue Min', 'Trackbars')
#     h_max = cv2.getTrackbarPos('Hue Max', 'Trackbars')
#     s_min = cv2.getTrackbarPos('Sat Min', 'Trackbars')
#     s_max = cv2.getTrackbarPos('Sat Max', 'Trackbars')
#     v_min = cv2.getTrackbarPos('Val Min', 'Trackbars')
#     v_max = cv2.getTrackbarPos('Val Max', 'Trackbars')
    
#     # 초기값 설정 또는 해제
#     if cv2.waitKey(1) & 0xFF == ord('s'):
#         if detection_enabled:
#             # 쓰레기 치우는 순간의 배경을 저장
#             post_cleanup_gray = frame_gray
#             cleanup_detection_enabled = True
#             print("Post-cleanup background set, ready for final detection.")
#         else:
#             # 초기 배경 설정
#             initial_gray = frame_gray
#             detection_enabled = True
#             print("Initial background set, detection enabled.")
    
#     # 검출 활성화된 경우에만 실행
#     if detection_enabled and initial_gray is not None and not cleanup_detection_enabled:
#         # 초기 이미지와 현재 이미지의 차이 계산
#         frame_delta = cv2.absdiff(initial_gray, frame_gray)
        
#         # 차이 이미지의 임계값 적용
#         _, diff_mask = cv2.threshold(frame_delta, diff_thresh, 255, cv2.THRESH_BINARY)
#         diff_mask = cv2.dilate(diff_mask, None, iterations=2)
#         cv2.imshow('diff_mask',diff_mask)

#         # Canny 엣지 검출
#         edges = cv2.Canny(frame_gray, threshold1, threshold2)
#         edges = cv2.dilate(edges, None, iterations=1)
#         cv2.imshow('edges1',edges)
#         edges = cv2.bitwise_and(edges, edges, mask=diff_mask)
#         cv2.imshow('edges2',edges)

#         # HSV 색상 범위에 따른 마스크 생성
#         lower_bound = np.array([h_min, s_min, v_min])
#         upper_bound = np.array([h_max, s_max, v_max])
#         hsv_mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

#         # 탐지 영역 마스크 생성 (마커와 별 인식 영역 제외)
#         detection_mask = np.zeros(diff_mask.shape, dtype=np.uint8)
#         detection_mask[roi_y_large:roi_y_large + roi_height_large, roi_x_large:roi_x_large + roi_width_large] = 255
#         detection_mask[roi_y_medium:roi_y_medium + roi_height_medium, roi_x_medium:roi_x_medium + roi_width_medium] = 0
#         detection_mask[roi_y_small:roi_y_small + roi_height_small, roi_x_small:roi_x_small + roi_width_small] = 0

#         # 최종 마스크 적용
#         combined_mask = cv2.bitwise_or(diff_mask, edges)
#         combined_mask = cv2.bitwise_and(combined_mask, combined_mask, mask=detection_mask)
#         combined_mask = cv2.bitwise_and(combined_mask, hsv_mask)

#         # 윤곽선 검출
#         contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         detected_objects = []

#         for cnt in contours:
#             if cv2.contourArea(cnt) > 150:  # 최소 면적 기준
#                 x, y, w, h = cv2.boundingRect(cnt)
#                 # 탐지된 객체가 지정된 탐지 영역 내에 있는지 확인
#                 if (roi_x_large <= x <= roi_x_large + roi_width_large and
#                     roi_y_large <= y <= roi_y_large + roi_height_large):
#                     detected_objects.append((x, y, w, h))
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                     cv2.putText(frame, 'Trash Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#         # 쓰레기가 계속 존재하는지 표시
#         if detected_objects:
#             cv2.putText(frame, 'Trash Present', (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

#         # 원본 프레임과 결합된 마스크를 나란히 보여줌
#         combined_frame = np.hstack((frame, cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)))

#         # 프레임을 화면에 표시
#         cv2.imshow('Original and Combined Mask', combined_frame)
#     # 'q' 키를 누르면 루프를 종료
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

   

# # 리소스 해제
# cap.release()
# cv2.destroyAllWindows()
