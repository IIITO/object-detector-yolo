import cv2
from ultralytics import YOLO

model = YOLO("yolov8x.pt")
model.to("cuda")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print(model.device)
print("FOURCC:", cap.get(cv2.CAP_PROP_FOURCC))
print("Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if not cap.isOpened():
    print("Не удалось открыть камеру")
    exit()

print("Запущено. Нажмите 'q' в окне видео, чтобы выйти.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить кадр")
        break

    results = model(frame, device=0, conf=0.1, verbose=False)

    annotated_frame = results[0].plot()

    cv2.imshow("Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
