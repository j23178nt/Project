import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path="blaze_face_short_range.tflite"),
    running_mode=VisionRunningMode.IMAGE,
)
detector = FaceDetector.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("エラー: カメラを開けませんでした")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("エラー: フレームを取得できませんでした")
        break

    # OpenCVはBGR、MediaPipeはRGBなので変換してから渡す
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    for detection in result.detections:
        bbox = detection.bounding_box
        # bounding_boxはfloatで返るのでcv2用にintへ変換
        x, y = int(bbox.origin_x), int(bbox.origin_y)
        w, h = int(bbox.width), int(bbox.height)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
