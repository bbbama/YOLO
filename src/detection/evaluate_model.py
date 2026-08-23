import os
import glob
import logging
import numpy as np
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Parametry
MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "train2" / "weights" / "best.pt"
IMAGES_DIR = PROJECT_ROOT / "data" / "lego_dataset" / "images" / "test"
LABELS_DIR = PROJECT_ROOT / "data" / "lego_dataset" / "labels" / "test"
CONF_THRESHOLD = 0.7
IOU_THRESHOLD = 0.5         # standard branżowy
ACCEPT_MAP_MIN = 0.5        # minimalny mAP@0.5 do akceptacji
ACCEPT_COUNT_ACC_MIN = 80.0 # minimalna zgodność liczby obiektów [%]


def load_gt_labels(label_path: str) -> np.ndarray:
    """
    Wczytuje etykiety YOLO z pliku tekstowego.

    Returns:
        Array shape (N, 5): [class, x_center, y_center, width, height]
        lub pusty array jeśli plik nie istnieje.
    """
    if not os.path.exists(label_path):
        return np.zeros((0, 5), dtype=float)
    try:
        with open(label_path, "r") as f:
            lines = [line.strip().split() for line in f if line.strip()]
        return np.array(lines, dtype=float) if lines else np.zeros((0, 5), dtype=float)
    except (ValueError, OSError) as e:
        logger.warning("Nie można wczytać etykiety %s: %s", label_path, e)
        return np.zeros((0, 5), dtype=float)


def yolo_to_corners(boxes: np.ndarray) -> np.ndarray:
    """
    Konwertuje boxy z formatu YOLO [cls, x_c, y_c, w, h]
    do formatu narożnikowego [x1, y1, x2, y2].
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, 4), dtype=float)
    x_c, y_c, w, h = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    return np.stack([x_c - w / 2, y_c - h / 2, x_c + w / 2, y_c + h / 2], axis=1)


def compute_iou(box: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """
    Oblicza IoU między jednym boxem predykcji a wszystkimi boxami GT.

    Args:
        box:      shape (4,)  — [x1, y1, x2, y2]
        gt_boxes: shape (N,4) — [x1, y1, x2, y2]

    Returns:
        Array shape (N,) z wartościami IoU ∈ [0, 1].
    """
    inter_x1 = np.maximum(box[0], gt_boxes[:, 0])
    inter_y1 = np.maximum(box[1], gt_boxes[:, 1])
    inter_x2 = np.minimum(box[2], gt_boxes[:, 2])
    inter_y2 = np.minimum(box[3], gt_boxes[:, 3])

    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union_area = box_area + gt_areas - inter_area

    return np.where(union_area > 0, inter_area / union_area, 0.0)


def compute_average_precision(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float,
) -> float:
    """
    Oblicza Average Precision (AP) dla jednego obrazu metodą 11-punktową.

    Predykcje sortowane są malejąco po score, następnie klasyfikowane
    jako TP/FP na podstawie IoU z GT (każdy GT może być dopasowany
    tylko raz - greedy matching).
    """
    n_gt = gt_boxes.shape[0]
    if n_gt == 0 and pred_boxes.shape[0] == 0:
        return 1.0  # brak obiektów i brak predykcji — ideał
    if n_gt == 0 or pred_boxes.shape[0] == 0:
        return 0.0

    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]

    gt_matched = np.zeros(n_gt, dtype=bool)
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))

    for i, p_box in enumerate(pred_boxes):
        ious = compute_iou(p_box, gt_boxes)
        best_gt = int(np.argmax(ious))

        if ious[best_gt] >= iou_threshold and not gt_matched[best_gt]:
            tp[i] = 1
            gt_matched[best_gt] = True
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    precision = cum_tp / (cum_tp + cum_fp + 1e-9)
    recall = cum_tp / (n_gt + 1e-9)

    # Pole pod krzywą P-R (metoda trapezoidalna)
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[1.0], precision, [0.0]])
    return float(np.trapezoid(precision, recall))


def main() -> None:
    if not os.path.exists(MODEL_PATH):
        logger.error("Nie znaleziono modelu: %s", MODEL_PATH)
        return

    model = YOLO(MODEL_PATH)

    image_files = glob.glob(os.path.join(IMAGES_DIR, "*.png"))
    if not image_files:
        logger.error("Nie znaleziono obrazów w: %s", IMAGES_DIR)
        return

    total_images = len(image_files)
    logger.info("Ewaluacja %d obrazów (conf=%.2f, IoU=%.2f)...",
                total_images, CONF_THRESHOLD, IOU_THRESHOLD)

    ap_scores: list[float] = []
    correct_count_images = 0

    for img_path in image_files:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(LABELS_DIR, f"{stem}.txt")

        gt_labels = load_gt_labels(label_path)
        gt_boxes = yolo_to_corners(gt_labels)

        results = model.predict(img_path, conf=CONF_THRESHOLD, verbose=False)[0]
        pred_boxes = results.boxes.xyxyn.cpu().numpy()
        pred_scores = results.boxes.conf.cpu().numpy()

        if len(pred_boxes) == len(gt_boxes):
            correct_count_images += 1

        ap = compute_average_precision(pred_boxes, pred_scores, gt_boxes, IOU_THRESHOLD)
        ap_scores.append(ap)

    mean_ap = float(np.mean(ap_scores)) if ap_scores else 0.0
    count_accuracy = (correct_count_images / total_images) * 100

    separator = "-" * 40
    logger.info(separator)
    logger.info("WYNIK EWALUACJI")
    logger.info("mAP@%.1f:                  %.4f", IOU_THRESHOLD, mean_ap)
    logger.info("Zgodność liczby obiektów: %.2f%% (%d/%d)",
                count_accuracy, correct_count_images, total_images)

    is_acceptable = mean_ap >= ACCEPT_MAP_MIN and count_accuracy >= ACCEPT_COUNT_ACC_MIN
    status = "AKCEPTOWALNA ✓" if is_acceptable else "DO POPRAWY ✗"
    logger.info("STATUS: %s", status)
    logger.info(separator)


if __name__ == "__main__":
    main()