import math
import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# MediaPipe Face Meshにおける左右の目のランドマーク番号（6点ずつ）
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.20  # これ未満なら目を閉じていると判定


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_ear(points):
    """Eye Aspect Ratio。値が小さいほど目を閉じている。"""
    vertical1 = distance(points[1], points[5])
    vertical2 = distance(points[2], points[4])
    horizontal = distance(points[0], points[3])

    if horizontal == 0:
        return 0.0
    return (vertical1 + vertical2) / (2.0 * horizontal)


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
)
landmarker = FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("エラー: カメラを開けませんでした")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("エラー: フレームを取得できませんでした")
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect(mp_image)

    if result.face_landmarks:
        face_landmarks = result.face_landmarks[0]

        left_points = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in LEFT_EYE]
        right_points = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in RIGHT_EYE]

        left_ear = calculate_ear(left_points)
        right_ear = calculate_ear(right_points)
        ear = (left_ear + right_ear) / 2.0

        status = "CLOSED" if ear < EAR_THRESHOLD else "OPEN"
        color = (0, 0, 255) if status == "CLOSED" else (0, 255, 0)

        for p in left_points + right_points:
            cv2.circle(frame, p, 3, color, -1)

        cv2.putText(frame, f"EAR: {ear:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Status: {status}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("EAR Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
