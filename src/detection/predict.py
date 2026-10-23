"""
Detekcja i liczenie klocków na zdjęciu lub całym folderze zdjęć.

Użycie:
    python src/detection/predict.py data/zdjecia [--model models/best.pt] [--conf 0.5] [--pokaz]
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    parser = argparse.ArgumentParser(description="Detekcja i liczenie klocków LEGO.")
    parser.add_argument("zrodlo", type=Path, help="zdjęcie lub folder ze zdjęciami")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "best.pt")
    parser.add_argument("--conf", type=float, default=0.5, help="próg pewności")
    parser.add_argument("--pokaz", action="store_true", help="wyświetl zdjęcia z bboxami")
    args = parser.parse_args()

    model = YOLO(args.model)
    results = model(args.zrodlo, conf=args.conf, stream=True, verbose=False)

    for r in results:
        print(f"{Path(r.path).name}: {len(r.boxes)} klocków")
        for box in r.boxes:
            print(f"  Pewność: {float(box.conf):.2f}, BBox: {box.xyxy[0].tolist()}")
        if args.pokaz:
            r.show()


if __name__ == "__main__":
    main()
