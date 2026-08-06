import math
import cv2  # OpenCV: カメラ映像の取得・描画処理用
import mediapipe as mp  # MediaPipe: 顔のランドマーク検出用

# --- MediaPipe Tasks API の設定 ---
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# MediaPipe Face Mesh における左右の目のランドマークインデックス (計6点ずつ)
# 配置順: [p1(左端), p2(上1), p3(上2), p4(右端), p5(下2), p6(下1)]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# 瞬き判定のしきい値 (EARがこの値未満なら「閉じている」と判定)
EAR_THRESHOLD = 0.20


# 2点間のユーグリッド距離を計算する関数
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Eye Aspect Ratio (EAR) を計算する関数
def calculate_ear(points):
    # points: [p1, p2, p3, p4, p5, p6]
    # 垂直方向のふたつの距離を計算
    vertical1 = distance(points[1], points[5])
    vertical2 = distance(points[2], points[4])

    # 水平方向の距離を計算
    horizontal = distance(points[0], points[3])

    # ゼロ除算 (ZeroDivisionError) を防止するための安全対策
    if horizontal == 0:
        return 0.0

    # EARの公式: (v1 + v2) / (2.0 * h)
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear


# FaceLandmarker のオプション設定
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,  # 静止画フレーム単位処理
    num_faces=1,  # 検出対象は1名のみ
)

# 検出器のインスタンス化
landmarker = FaceLandmarker.create_from_options(options)

# Webカメラの初期化
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("エラー: カメラを開くことができませんでした。")
    exit()

# リアルタイム処理ループ
while True:
    ret, frame = cap.read()

    if not ret:
        print("エラー: フレームを取得できませんでした。")
        break

    h, w, _ = frame.shape  # フレームの解像度を取得

    # BGRからRGBに変換し、MediaPipe専用の画像形式に変換
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # ランドマーク検出を実行
    result = landmarker.detect(mp_image)

    if result.face_landmarks:
        face_landmarks = result.face_landmarks[0]

        left_points = []
        right_points = []

        # 左目のピクセル座標を取得
        for idx in LEFT_EYE:
            point = face_landmarks[idx]
            left_points.append((int(point.x * w), int(point.y * h)))

        # 右目のピクセル座標を取得
        for idx in RIGHT_EYE:
            point = face_landmarks[idx]
            right_points.append((int(point.x * w), int(point.y * h)))

        # 左右のEARを計算し、その平均値を求める
        left_ear = calculate_ear(left_points)
        right_ear = calculate_ear(right_points)
        ear = (left_ear + right_ear) / 2.0

        # 目の特徴点を画面に描画 (緑色の点)
        for p in left_points + right_points:
            cv2.circle(frame, p, 3, (0, 255, 0), -1)

        # EAR値をもとに目の開閉状態を判定 (閾値未満なら CLOSED)
        if ear < EAR_THRESHOLD:
            status = "CLOSED"
            color = (0, 0, 255)  # 閉じた時は赤色表示
        else:
            status = "OPEN"
            color = (0, 255, 0)  # 開いている時は緑色表示

        # 画面にEAR値と状態を出力
        cv2.putText(
            frame,
            f"EAR: {ear:.2f}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )
        cv2.putText(
            frame,
            f"Status: {status}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )

    # 画面描画の更新
    cv2.imshow("EAR Detection", frame)

    # 'q' キー押下で終了
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# リソース解放
cap.release()
cv2.destroyAllWindows()
landmarker.close()