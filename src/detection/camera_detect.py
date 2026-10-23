import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="Detekcja klocków na obrazie z kamery.")
    # "0" to zazwyczaj wbudowana kamera, "1" kolejna (np. USB / telefon)
    parser.add_argument("--source", default="1",
                        help="numer kamery lub ścieżka/URL strumienia")
    parser.add_argument("--model", type=Path,
                        default=PROJECT_ROOT / "models" / "best.pt")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="próg pewności")
    args = parser.parse_args()

    model = YOLO(args.model)

    print("Uruchamianie kamery... Naciśnij 'q', aby wyjść.")

    # stream=True pozwala na płynne przetwarzanie klatek
    results = model.predict(source=args.source, show=True, conf=args.conf,
                            stream=True)

    # Ta pętla musi istnieć, aby stream działał
    for r in results:
        pass


if __name__ == "__main__":
    main()
