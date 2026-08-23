from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

model = YOLO(PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt")

results = model(PROJECT_ROOT / "data" / "zdjecia" / "zdjecie_klocka3.jpg", conf=0.25)

# Wyświetl zdjęcie z bboxami
results[0].show()

# Dane liczbowe
for box in results[0].boxes:
    print(f"Klasa: {box.cls}, Pewność: {box.conf:.2f}, BBox: {box.xyxy}")