import math
import threading
import time
import cv2  # OpenCV：カメラ映像の取得、描画、画面表示に使用
import mediapipe as mp  # MediaPipe：顔ランドマークの検出に使用
import pyttsx3  # テキストを音声として読み上げるために使用
import winsound  # Windowsのビープ音を再生するために使用


# ---------------------------------------------------------------------------
# MediaPipe Tasks APIのクラス設定
# ---------------------------------------------------------------------------

# モデルファイルのパスなど、基本的な設定に使用する
BaseOptions = mp.tasks.BaseOptions

# 顔のランドマーク検出を実行するためのクラス
FaceLandmarker = mp.tasks.vision.FaceLandmarker

# 使用するモデル、実行モード、検出人数などを設定するためのクラス
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

# IMAGE、VIDEO、LIVE_STREAMなどの実行モードを指定する
VisionRunningMode = mp.tasks.vision.RunningMode


# ---------------------------------------------------------------------------
# 顔ランドマーク番号
# ---------------------------------------------------------------------------

# EARの計算に使用する左目周辺の6点
LEFT_EYE = [33, 160, 158, 133, 153, 144]

# EARの計算に使用する右目周辺の6点
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# MARの計算に使用する口周辺のランドマーク
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 78
RIGHT_MOUTH = 308


# ---------------------------------------------------------------------------
# 判定に使用する閾値
# ---------------------------------------------------------------------------

# EARが0.20未満の場合、目が閉じていると判定する
EAR_THRESHOLD = 0.20

# 目を閉じた状態が45フレームを超えた場合、
# 居眠りの可能性があると判定する
DROWSY_FRAMES = 45

# MARが0.35を超えた場合、口が大きく開いていると判定する
MAR_THRESHOLD = 0.35

# 口が開いた状態が30フレームを超えた場合、
# あくびと判定する
YAWN_FRAMES = 30

# PERCLOSが20％を超えた場合、疲労状態と判定する
PERCLOS_TIRED = 20

# PERCLOSが40％を超えた場合、居眠り状態と判定する
PERCLOS_DROWSY = 40

# 音声警告を繰り返す間隔（秒）
WARNING_INTERVAL = 5

# 最後に警告を行った時刻
last_warning_time = 0


# ---------------------------------------------------------------------------
# フレーム数と状態を管理する変数
# ---------------------------------------------------------------------------

# 目を連続して閉じているフレーム数
closed_frames = 0

# 口を連続して開いているフレーム数
yawn_frames = 0

# 顔を検出できた全フレーム数
total_frames = 0

# 目が閉じていたフレームの合計数
closed_eye_frames = 0

# 音声警告が現在再生中かどうか
alarm_playing = False


# ---------------------------------------------------------------------------
# 音声警告処理
# ---------------------------------------------------------------------------

def warning_alarm():
    """
    ビープ音と音声メッセージを再生する。

    音声警告中に新しい警告処理が開始されないよう、
    alarm_playingで再生状態を管理する。
    """
    global alarm_playing

    if alarm_playing:
        return

    alarm_playing = True

    try:
        # 音声メッセージの前にビープ音を再生する
        winsound.Beep(2000, 300)

        # スレッド内で音声エンジンを生成する
        local_engine = pyttsx3.init()

        # 読み上げ速度を設定する
        local_engine.setProperty("rate", 150)

        # 音量を最大に設定する
        local_engine.setProperty("volume", 1.0)

        # 警告メッセージを読み上げる
        local_engine.say(
            "居眠り運転の危険があります。休憩してください"
        )
        local_engine.runAndWait()
        local_engine.stop()

        # 音声メッセージの後にビープ音を再生する
        winsound.Beep(2000, 300)

    except Exception as error:
        print(f"音声出力エラーが発生しました: {error}")

    finally:
        alarm_playing = False


# ---------------------------------------------------------------------------
# 2点間の距離計算
# ---------------------------------------------------------------------------

def distance(p1, p2):
    """
    2点間のユークリッド距離を計算する。
    """
    return math.sqrt(
        (p1[0] - p2[0]) ** 2
        + (p1[1] - p2[1]) ** 2
    )


# ---------------------------------------------------------------------------
# EARの計算
# ---------------------------------------------------------------------------

def calculate_ear(points):
    """
    目周辺の6点からEARを計算する。

    目が開いている場合は縦方向の距離が大きくなり、
    EARの値も高くなる。

    目が閉じている場合は縦方向の距離が小さくなり、
    EARの値も低くなる。
    """
    vertical1 = distance(
        points[1],
        points[5]
    )

    vertical2 = distance(
        points[2],
        points[4]
    )

    horizontal = distance(
        points[0],
        points[3]
    )

    # 横方向の距離が0の場合に発生するゼロ除算を防ぐ
    if horizontal == 0:
        return 0.0

    return (
        vertical1 + vertical2
    ) / (2.0 * horizontal)


# ---------------------------------------------------------------------------
# MARの計算
# ---------------------------------------------------------------------------

def calculate_mar(face_landmarks, w, h):
    """
    口周辺の4点からMARを計算する。

    口の縦方向の距離を横方向の距離で割ることで、
    口の開き具合を数値化する。
    """
    upper = face_landmarks[UPPER_LIP]
    lower = face_landmarks[LOWER_LIP]
    left = face_landmarks[LEFT_MOUTH]
    right = face_landmarks[RIGHT_MOUTH]

    # MediaPipeの正規化座標をピクセル座標へ変換する
    upper_p = (
        int(upper.x * w),
        int(upper.y * h)
    )

    lower_p = (
        int(lower.x * w),
        int(lower.y * h)
    )

    left_p = (
        int(left.x * w),
        int(left.y * h)
    )

    right_p = (
        int(right.x * w),
        int(right.y * h)
    )

    vertical = distance(
        upper_p,
        lower_p
    )

    horizontal = distance(
        left_p,
        right_p
    )

    # 横方向の距離が0の場合に発生するゼロ除算を防ぐ
    if horizontal == 0:
        return (
            0.0,
            upper_p,
            lower_p,
            left_p,
            right_p
        )

    mar = vertical / horizontal

    return (
        mar,
        upper_p,
        lower_p,
        left_p,
        right_p
    )


# ---------------------------------------------------------------------------
# Face Landmarkerの初期化
# ---------------------------------------------------------------------------

options = FaceLandmarkerOptions(
    # このPythonファイルと同じフォルダにあるモデルを使用する
    base_options=BaseOptions(
        model_asset_path="face_landmarker.task"
    ),

    # Webカメラの各フレームを独立した画像として処理する
    running_mode=VisionRunningMode.IMAGE,

    # 本研究では運転者1名のみを検出対象とする
    num_faces=1,
)

# 設定したオプションを使用してFace Landmarkerを生成する
landmarker = FaceLandmarker.create_from_options(options)


# ---------------------------------------------------------------------------
# Webカメラの起動
# ---------------------------------------------------------------------------

# 0はPCに接続されている標準Webカメラを表す
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("エラー: カメラを開くことができませんでした。")
    landmarker.close()
    raise SystemExit


# ---------------------------------------------------------------------------
# メインループ
# ---------------------------------------------------------------------------

while True:
    # Webカメラから1フレーム取得する
    ret, frame = cap.read()

    if not ret:
        print(
            "エラー: カメラから画像を取得できませんでした。"
        )
        break

    # 取得した画像の高さと幅を取得する
    h, w, _ = frame.shape

    # OpenCVはBGR形式、MediaPipeはRGB形式を使用するため変換する
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # OpenCV画像をMediaPipeで処理可能な形式へ変換する
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # 顔ランドマークの検出を実行する
    result = landmarker.detect(mp_image)

    if result.face_landmarks:
        # num_faces=1のため、最初に検出された顔を使用する
        face_landmarks = result.face_landmarks[0]

        left_points = []
        right_points = []

        # 左目周辺のランドマークをピクセル座標へ変換する
        for idx in LEFT_EYE:
            point = face_landmarks[idx]

            left_points.append(
                (
                    int(point.x * w),
                    int(point.y * h)
                )
            )

        # 右目周辺のランドマークをピクセル座標へ変換する
        for idx in RIGHT_EYE:
            point = face_landmarks[idx]

            right_points.append(
                (
                    int(point.x * w),
                    int(point.y * h)
                )
            )

        # 左右の目のEARを計算する
        left_ear = calculate_ear(left_points)
        right_ear = calculate_ear(right_points)

        # 左右の目の平均EARを使用する
        ear = (left_ear + right_ear) / 2.0

        # 口のMARと描画用の座標を取得する
        (
            mar,
            upper_p,
            lower_p,
            left_mouth_p,
            right_mouth_p
        ) = calculate_mar(
            face_landmarks,
            w,
            h
        )

        # 顔を検出できたフレーム数を加算する
        total_frames += 1

        # EARが閾値未満の場合、目が閉じていると判定する
        if ear < EAR_THRESHOLD:
            closed_frames += 1
            closed_eye_frames += 1

        else:
            # 目が開いた場合、連続閉眼フレーム数をリセットする
            closed_frames = 0

        # MARが閾値を超えた場合、口が開いていると判定する
        if mar > MAR_THRESHOLD:
            yawn_frames += 1

        else:
            # 口が閉じた場合、連続フレーム数をリセットする
            yawn_frames = 0

        # 目が閉じていたフレームの割合をPERCLOSとして計算する
        perclos = (
            closed_eye_frames / total_frames
        ) * 100

        # 口が一定時間以上開いている場合、あくびと判定する
        is_yawning = yawn_frames > YAWN_FRAMES

        # ---------------------------------------------------------------
        # 運転者の状態判定
        # ---------------------------------------------------------------

        if (
            closed_frames > DROWSY_FRAMES
            or perclos > PERCLOS_DROWSY
        ):
            # 長時間の閉眼、またはPERCLOSが高い場合
            status = "DROWSY"
            color = (0, 0, 255)

        elif (
            is_yawning
            or perclos > PERCLOS_TIRED
        ):
            # あくび、またはPERCLOSの上昇を検出した場合
            status = "TIRED / YAWNING"
            color = (0, 165, 255)

        elif ear < EAR_THRESHOLD:
            # 短時間だけ目を閉じている場合
            status = "CLOSED"
            color = (0, 255, 255)

        else:
            # 通常の覚醒状態
            status = "OPEN"
            color = (0, 255, 0)

        # ---------------------------------------------------------------
        # 音声警告
        # ---------------------------------------------------------------

        if status in [
            "DROWSY",
            "TIRED / YAWNING"
        ]:
            current_time = time.time()

            # 前回の警告から指定時間が経過した場合のみ再生する
            if (
                current_time - last_warning_time
                > WARNING_INTERVAL
            ):
                last_warning_time = current_time

                # 音声再生中もカメラ処理を継続するため、
                # 別スレッドで警告処理を実行する
                threading.Thread(
                    target=warning_alarm,
                    daemon=True
                ).start()

        # ---------------------------------------------------------------
        # 目周辺の描画
        # ---------------------------------------------------------------

        for point in left_points + right_points:
            cv2.circle(
                frame,
                point,
                3,
                color,
                -1
            )

        # ---------------------------------------------------------------
        # 口周辺の描画
        # ---------------------------------------------------------------

        cv2.circle(
            frame,
            upper_p,
            4,
            color,
            -1
        )

        cv2.circle(
            frame,
            lower_p,
            4,
            color,
            -1
        )

        cv2.circle(
            frame,
            left_mouth_p,
            4,
            color,
            -1
        )

        cv2.circle(
            frame,
            right_mouth_p,
            4,
            color,
            -1
        )

        # 口の縦方向と横方向の距離を線で表示する
        cv2.line(
            frame,
            upper_p,
            lower_p,
            color,
            2
        )

        cv2.line(
            frame,
            left_mouth_p,
            right_mouth_p,
            color,
            2
        )

        # ---------------------------------------------------------------
        # 判定結果の表示
        # ---------------------------------------------------------------

        cv2.putText(
            frame,
            f"EAR: {ear:.2f}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.putText(
            frame,
            f"MAR: {mar:.2f}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.putText(
            frame,
            f"Status: {status}",
            (30, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.putText(
            frame,
            f"Closed Frames: {closed_frames}",
            (30, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.putText(
            frame,
            f"Yawn Frames: {yawn_frames}",
            (30, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.putText(
            frame,
            f"PERCLOS: {perclos:.1f}%",
            (30, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    # 処理結果を画面に表示する
    cv2.imshow(
        "Drowsiness Detection",
        frame
    )

    # Qキーが押された場合はプログラムを終了する
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ---------------------------------------------------------------------------
# リソースの解放
# ---------------------------------------------------------------------------

cap.release()
cv2.destroyAllWindows()
landmarker.close()