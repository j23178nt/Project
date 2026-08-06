import cv2  # OpenCVライブラリをインポート（カメラ操作や画像処理に使用）

# Webカメラの初期化 (0はデフォルトの内蔵カメラまたは最初に認識されたWebカメラを指定)
cap = cv2.VideoCapture(0)

# カメラが正常に開けたか確認
if not cap.isOpened():
    print("エラー: カメラを開くことができませんでした。")
    exit()

# リアルタイムでフレームを取得・表示するためのメインループ
while True:
    # カメラから1フレーム読み込む
    # ret  -> 取得成功ならTrue、失敗ならFalse
    # frame -> 取得した画像データ (NumPy配列)
    ret, frame = cap.read()

    # フレーム取得に失敗した場合はループを抜ける
    if not ret:
        print("エラー: カメラから画像を取得できませんでした。")
        break

    # 取得したフレームをウィンドウに表示
    cv2.imshow("Camera Test", frame)

    # キー入力を1ms待機
    # 'q' キーが押されたらループを終了してプログラムを抜ける
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- リソースの解放処理 ---
cap.release()  # カメラデバイスを解放
cv2.destroyAllWindows()  # 開いたすべてのOpenCVウィンドウを閉じる