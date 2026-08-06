import cv2  # カメラ映像の取得や画像処理のために使用
import mediapipe as mp  # 顔検出や特徴点抽出（ランドマーク）を行うAIライブラリ

# MediaPipeのタスクAPIから必要なクラスを読み込む
BaseOptions = mp.tasks.BaseOptions  # モデルの基本設定（.tfliteファイルの me/パスなど）
FaceDetector = mp.tasks.vision.FaceDetector  # 顔検出を実行するメインクラス
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions  # 検出オプションの設定クラス
VisionRunningMode = mp.tasks.vision.RunningMode  # 実行モードの設定（IMAGE, VIDEO, LIVE_STREAM）

# --- FaceDetectorの設定 ---
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path="blaze_face_short_range.tflite"),
    running_mode=VisionRunningMode.IMAGE,  # 静止画モードで処理
)

# オプションをもとに検出器を作成
detector = FaceDetector.create_from_options(options)

# --- Webカメラの初期化 ---
cap = cv2.VideoCapture(0)  # 0: 内蔵カメラまたは標準のWebカメラ

if not cap.isOpened():
    print("エラー: カメラを開くことができませんでした。")
    exit()

# --- リアルタイム処理ループ ---
while True:
    ret, frame = cap.read()  # ret: 読み込み成功フラグ, frame: カメラの画像データ

    if not ret:
        print("エラー: フレームを取得できませんでした。")
        break

    # OpenCVはBGR形式、MediaPipeはRGB形式を使用するため色空間を変換
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe専用の画像フォーマット（Imageオブジェクト）に変換
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # AIモデルに画像を入力して顔を検出
    result = detector.detect(mp_image)

    # 検出された各顔に対してバウンディングボックスを描画
    if result.detections:
        for detection in result.detections:
            bbox = detection.bounding_box

            # 【重要・修正箇所】
            # bounding_boxの値はfloat型で返されるため、OpenCVで描画するためにint型へキャストする
            x = int(bbox.origin_x)
            y = int(bbox.origin_y)
            w = int(bbox.width)
            h = int(bbox.height)

            # 顔を囲む緑色の矩形を描画 (画像, 左上座標, 右下座標, 色(BGR), 線の太さ)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 検出テキストを表示
            cv2.putText(
                frame,
                "Face",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

    # 画面に描画結果を表示
    cv2.imshow("Face Detection", frame)

    # 'q' キーが押されたらループを抜け出す (1ms待機)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- リソースの解放処理 ---
cap.release()  # カメラを解放
cv2.destroyAllWindows()  # 表示ウィンドウをすべて閉じる
detector.close()  # MediaPipeの検出器を終了