import math
import threading
import time
from collections import deque
import cv2          # カメラ映像の取得・描画・表示
import mediapipe as mp  # 顔ランドマーク検出
import pyttsx3       # 音声読み上げ
import winsound      # Windowsのビープ音

# --- MediaPipe Tasks API の準備 ---
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# EAR計算に使う目のランドマーク番号（MediaPipe FaceMeshの点番号）
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# MAR計算に使う口のランドマーク番号
UPPER_LIP, LOWER_LIP = 13, 14
LEFT_MOUTH, RIGHT_MOUTH = 78, 308

# 判定用の閾値。何度か試して以下の値に落ち着いた
EAR_THRESHOLD = 0.20      # これ未満なら目を閉じていると判定
DROWSY_FRAMES = 45        # 閉眼がこのフレーム数を超えたら危険
MAR_THRESHOLD = 0.35      # これを超えたら口が開いていると判定
YAWN_FRAMES = 30          # あくびと判定するフレーム数
PERCLOS_TIRED = 20        # 疲労の目安(%)
PERCLOS_DROWSY = 40       # 危険水準の目安(%)
WARNING_INTERVAL = 5      # 音声警告を繰り返す最小間隔(秒)

last_warning_time = 0
closed_frames = 0  # 連続で目を閉じているフレーム数
yawn_frames = 0     # 連続であくびしているフレーム数
alarm_playing = False

# PERCLOSは「直近一定時間」に対する閉眼割合で計算する必要があるため、
# 全フレームを累積するのではなく、直近PERCLOS_WINDOW_SECONDS秒分だけを
# dequeに保持して算出する（古いフレームは自動で捨てられる）
PERCLOS_WINDOW_SECONDS = 30
ASSUMED_FPS = 15  # 環境により実際のFPSと差が出るため目安値
PERCLOS_WINDOW_SIZE = PERCLOS_WINDOW_SECONDS * ASSUMED_FPS
eye_state_window = deque(maxlen=PERCLOS_WINDOW_SIZE)  # 1: 閉眼, 0: 開眼


def warning_alarm():
    """ビープ音＋音声で警告を出す。多重再生を防ぐためalarm_playingで管理。"""
    global alarm_playing
    if alarm_playing:
        return

    alarm_playing = True
    try:
        winsound.Beep(2000, 300)

        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.setProperty("volume", 1.0)
        engine.say("居眠り運転の危険があります。休憩してください")
        engine.runAndWait()
        engine.stop()

        winsound.Beep(2000, 300)
    except Exception as e:
        print(f"音声出力エラー: {e}")
    finally:
        alarm_playing = False


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_ear(points):
    """目の開閉度(EAR)を計算。値が小さいほど目を閉じている。"""
    vertical1 = distance(points[1], points[5])
    vertical2 = distance(points[2], points[4])
    horizontal = distance(points[0], points[3])

    if horizontal == 0:
        return 0.0
    return (vertical1 + vertical2) / (2.0 * horizontal)


def calculate_mar(landmarks, w, h):
    """口の開き具合(MAR)と描画用座標を返す。"""
    def to_px(idx):
        p = landmarks[idx]
        return int(p.x * w), int(p.y * h)

    upper_p, lower_p = to_px(UPPER_LIP), to_px(LOWER_LIP)
    left_p, right_p = to_px(LEFT_MOUTH), to_px(RIGHT_MOUTH)

    horizontal = distance(left_p, right_p)
    if horizontal == 0:
        return 0.0, upper_p, lower_p, left_p, right_p

    mar = distance(upper_p, lower_p) / horizontal
    return mar, upper_p, lower_p, left_p, right_p


# --- Face Landmarker の初期化 ---
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,  # 運転者1人だけを想定
)
landmarker = FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("エラー: カメラを開けませんでした")
    landmarker.close()
    raise SystemExit


while True:
    ret, frame = cap.read()
    if not ret:
        print("エラー: フレームを取得できませんでした")
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    face_landmarks = result.face_landmarks[0]

    left_points = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in LEFT_EYE]
    right_points = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in RIGHT_EYE]

    left_ear = calculate_ear(left_points)
    right_ear = calculate_ear(right_points)
    ear = (left_ear + right_ear) / 2.0

    mar, upper_p, lower_p, left_mouth_p, right_mouth_p = calculate_mar(face_landmarks, w, h)

    is_eye_closed = ear < EAR_THRESHOLD
    eye_state_window.append(1 if is_eye_closed else 0)

    if is_eye_closed:
        closed_frames += 1
    else:
        closed_frames = 0

    yawn_frames = yawn_frames + 1 if mar > MAR_THRESHOLD else 0

    # 直近PERCLOS_WINDOW_SECONDS秒間における閉眼フレームの割合
    perclos = (sum(eye_state_window) / len(eye_state_window)) * 100
    is_yawning = yawn_frames > YAWN_FRAMES

    # 状態判定（危険度が高い順にチェック）
    if closed_frames > DROWSY_FRAMES or perclos > PERCLOS_DROWSY:
        status, color = "DROWSY", (0, 0, 255)
    elif is_yawning or perclos > PERCLOS_TIRED:
        status, color = "TIRED / YAWNING", (0, 165, 255)
    elif ear < EAR_THRESHOLD:
        status, color = "CLOSED", (0, 255, 255)
    else:
        status, color = "OPEN", (0, 255, 0)

    # 危険/疲労状態なら音声警告（間隔を空けて再生）
    if status in ("DROWSY", "TIRED / YAWNING"):
        now = time.time()
        if now - last_warning_time > WARNING_INTERVAL:
            last_warning_time = now
            threading.Thread(target=warning_alarm, daemon=True).start()

    # --- 描画 ---
    for point in left_points + right_points:
        cv2.circle(frame, point, 3, color, -1)

    for p in (upper_p, lower_p, left_mouth_p, right_mouth_p):
        cv2.circle(frame, p, 4, color, -1)

    cv2.line(frame, upper_p, lower_p, color, 2)
    cv2.line(frame, left_mouth_p, right_mouth_p, color, 2)

    info_lines = [
        f"EAR: {ear:.2f}",
        f"MAR: {mar:.2f}",
        f"Status: {status}",
        f"Closed Frames: {closed_frames}",
        f"Yawn Frames: {yawn_frames}",
        f"PERCLOS: {perclos:.1f}%",
    ]
    for i, text in enumerate(info_lines):
        cv2.putText(frame, text, (30, 50 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
landmarker.close()
