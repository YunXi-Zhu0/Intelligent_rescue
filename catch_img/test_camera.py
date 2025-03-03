import cv2


#用于测试摄像头是否正常
cap = cv2.VideoCapture(8)

if not cap.isOpened():
    print("无法打开摄像头!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧!")
        break
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
