import time
import cv2  # OpenCV: カメラ映像の取得および描画処理用
import mediapipe as mp  # MediaPipe: 顔のランドマーク（特徴点）検出用

# MediaPipe Tasks API のクラスを定義
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# MediaPipe Face Mesh における左右の目のランドマークインデックス定義
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# --- LIVE_STREAM モード用の設定 ---
# 非同期処理（LIVE_STREAM）用のオプション設定
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,  # リアルタイム映像処理モード
    num_faces=1,  # 検出する最大の顔の数
    # 非同期で結果を受け取るコールバック関数（必須）
    result_callback=lambda result, output_image, timestamp_ms: None,
)

# -------------------------------------------------------------------
# ※ 注意: LIVE_STREAM モードでの非同期処理をよりシンプルに同期実行（Direct Detect）
# したい場合は、以下のように IMAGE モードのまま正しく座標変換を行います。
# 以下は安定動作する標準的なコード例です。
# -------------------------------------------------------------------

options_image = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
)

landmarker = FaceLandmarker.create_from_options(options_image)

# Webカメラの初期化 (0: 標準Webカメラ)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("エラー: カメラを開くことができませんでした。")
    exit()

# メイン処理ループ
while True:
    ret, frame = cap.read()  # カメラから1フレーム取得

    if not ret:
        print("エラー: フレームを取得できませんでした。")
        break

    # フレームの高さ(h)と幅(w)を取得（正規化座標をピクセル座標に変換するために使用）
    h, w, _ = frame.shape

    # OpenCVのBGR形式からMediaPipeが対応するRGB形式へ変換
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # 顔のランドマーク検出を実行
    result = landmarker.detect(mp_image)

    # 顔が検出された場合の処理
    if result.face_landmarks:
        # 最初の顔のランドマークリストを取得 (1人分のデータ)
        face_landmarks = result.face_landmarks[0]

        # 左右の目のインデックスのみをループ処理
        for idx in LEFT_EYE + RIGHT_EYE:
            point = face_landmarks[idx]

            # ランドマーク座標は0.0～1.0に正規化されているため、
            # 画像の幅(w)と高さ(h)を掛けて実際のピクセル座標(x, y)に変換する
            x = int(point.x * w)
            y = int(point.y * h)

            # 目のポイントに緑色の円を描画 (画像, 中心座標, 半径, 色(BGR), 塗りつぶし)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            # ポイントのインデックス番号をテキスト表示
            cv2.putText(
                frame,
                str(idx),
                (x + 3, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 255, 0),
                1,
            )

    # 描画結果を画面に表示
    cv2.imshow("Eye Landmarks", frame)

    # 'q' キーが押されたらループを終了 (1ms待機)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# リソースの解放処理
cap.release()  # カメラデバイスを解放
cv2.destroyAllWindows()  # 表示ウィンドウをすべて閉じる
landmarker.close()  # Landmarkerオブジェクトを破棄