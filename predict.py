from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

results = model("zdjecie_klocka3.jpg", conf=0.25)

# Wyświetl zdjęcie z bboxami
results[0].show()

# Dane liczbowe
for box in results[0].boxes:
    print(f"Klasa: {box.cls}, Pewność: {box.conf:.2f}, BBox: {box.xyxy}")