from ultralytics import YOLO

model = YOLO("yolo11m.pt")

results = model.predict('InputVideos/08fd33_4.mp4', save=True,stream=True)
print(results[0])
print('--------------------------------')
for box in results[0].boxes:
    print(box)
