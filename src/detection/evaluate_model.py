"""
Ewaluacja modelu na splicie test.

1. Standardowe metryki detekcji (ultralytics `model.val`, liczone na całym
   zbiorze): precision, recall, mAP@0.5, mAP@0.5:0.95.
2. Zgodność liczby obiektów — odsetek obrazów, na których liczba detekcji
   (po progu conf) równa się liczbie klocków w etykiecie. To bezpośrednia
   miara celu projektu (liczenie klocków).

Użycie:
    python src/detection/evaluate_model.py
        [--model models/best.pt] [--conf 0.5]
"""

import argparse
import glob
import logging
import os
from pathlib import Path

import yaml
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Parametry domyślne
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
DATA_YAML = PROJECT_ROOT / "data" / "lego_dataset" / "data.yaml"
CONF_THRESHOLD = 0.5        # próg pewności dla liczenia obiektów
IOU_THRESHOLD = 0.5         # standard branżowy
ACCEPT_MAP_MIN = 0.5        # minimalny mAP@0.5 do akceptacji
ACCEPT_COUNT_ACC_MIN = 80.0  # minimalna zgodność liczby obiektów [%]


def policz_etykiety(label_path: str) -> int:
    """Liczba obiektów w pliku etykiet YOLO (brak pliku = 0)."""
    if not os.path.exists(label_path):
        return 0
    with open(label_path, "r") as f:
        return sum(1 for line in f if line.strip())


def katalog_testu(data_yaml: Path) -> Path:
    """Katalog obrazów splitu test na podstawie data.yaml."""
    with open(data_yaml, "r") as f:
        cfg = yaml.safe_load(f)
    if "test" not in cfg:
        raise KeyError(f"Brak klucza 'test' w {data_yaml} — "
                       "wygeneruj dataset ponownie.")
    baza = Path(cfg.get("path") or data_yaml.parent)
    if not baza.is_absolute():
        baza = data_yaml.parent / baza
    return baza / cfg["test"]


def zgodnosc_liczby(model: YOLO, images_dir: Path,
                    conf: float) -> tuple[int, int]:
    """Zwraca (obrazy z poprawnie policzonymi klockami, wszystkie obrazy)."""
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    poprawne = 0
    for img_path in image_files:
        # images/test/x.png -> labels/test/x.txt
        label_path = img_path.replace(f"{os.sep}images{os.sep}",
                                      f"{os.sep}labels{os.sep}")
        label_path = os.path.splitext(label_path)[0] + ".txt"
        wynik = model.predict(img_path, conf=conf, verbose=False)[0]
        if len(wynik.boxes) == policz_etykiety(label_path):
            poprawne += 1
    return poprawne, len(image_files)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ewaluacja modelu YOLO na splicie test.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH,
                        help="ścieżka do wag .pt")
    parser.add_argument("--data", type=Path, default=DATA_YAML,
                        help="ścieżka do data.yaml")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD,
                        help="próg pewności dla liczenia obiektów")
    args = parser.parse_args()

    if not args.model.exists():
        logger.error("Nie znaleziono modelu: %s", args.model)
        return

    images_dir = katalog_testu(args.data)
    if not glob.glob(os.path.join(images_dir, "*.png")):
        logger.error("Nie znaleziono obrazów w: %s", images_dir)
        return

    model = YOLO(args.model)

    # 1. Standardowe metryki (conf niski — mAP liczony po całej krzywej P-R)
    metryki = model.val(data=str(args.data), split="test", iou=0.7,
                        plots=False, verbose=False,
                        project=str(PROJECT_ROOT / "runs" / "detect"),
                        name="eval_test", exist_ok=True)
    box = metryki.box

    # 2. Zgodność liczby obiektów
    poprawne, n_obrazow = zgodnosc_liczby(model, images_dir, args.conf)
    count_accuracy = poprawne / n_obrazow * 100

    separator = "-" * 40
    logger.info(separator)
    logger.info("WYNIK EWALUACJI (%d obrazów testowych)", n_obrazow)
    logger.info("Precision:                %.4f", box.mp)
    logger.info("Recall:                   %.4f", box.mr)
    logger.info("mAP@%.1f:                  %.4f", IOU_THRESHOLD, box.map50)
    logger.info("mAP@0.5:0.95:             %.4f", box.map)
    logger.info("Zgodność liczby obiektów: %.2f%% (%d/%d, conf=%.2f)",
                count_accuracy, poprawne, n_obrazow, args.conf)

    is_acceptable = (box.map50 >= ACCEPT_MAP_MIN
                     and count_accuracy >= ACCEPT_COUNT_ACC_MIN)
    status = "AKCEPTOWALNA ✓" if is_acceptable else "DO POPRAWY ✗"
    logger.info("STATUS: %s", status)
    logger.info(separator)


if __name__ == "__main__":
    main()
