from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def main():
    # Załaduj TWÓJ wytrenowany model
    model = YOLO(PROJECT_ROOT / "models" / "best.pt")
    #runs/detect/train2/weights/best.pt

    print("Uruchamianie kamery... Naciśnij 'q', aby wyjść.")
    
    # source="0" to zazwyczaj wbudowana kamera lub pierwsza na USB
    # imgsz=640 to standardowy rozmiar dla YOLO
    # stream=True pozwala na płynne przetwarzanie klatek
    results = model.predict(source="1", show=True, conf=0.4, stream=True)

    # Ta pętla musi istnieć, aby stream działał
    for r in results:
        pass

if __name__ == "__main__":
    main()
