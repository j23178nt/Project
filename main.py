import cv2

cap = cv2.VideoCapture(0)  # 0: 標準のWebカメラ

if not cap.isOpened():
    print("エラー: カメラを開けませんでした")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("エラー: フレームを取得できませんでした")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
