import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# MediaPipe Face Meshにおける左右の目のランドマーク番号
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

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

        for idx in LEFT_EYE + RIGHT_EYE:
            point = face_landmarks[idx]
            # 正規化座標(0.0〜1.0)を画像サイズに合わせてピクセル座標へ変換
            x, y = int(point.x * w), int(point.y * h)

            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            cv2.putText(frame, str(idx), (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    cv2.imshow("Eye Landmarks", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
