"""
Moduł augmentacji obrazów dla datasetu YOLO.
Augmentacja wyłącznie pikselowa — geometria i etykiety bez zmian.

Użycie (batch po splitcie train):
    python src/augmentuj.py data/lego_dataset [--kopie 1] [--seed 42]
Dla każdego obrazu powstaje kopia *_augK.png z identyczną labelką .txt.

Wymaga albumentations >= 2.0 (API std_range w GaussNoise).
"""

import argparse
import glob
import os
import random
import shutil
import zlib

import albumentations as A
import cv2
import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Parametry augmentacji
# =============================================================================

# Szum sensora
_SZUM_GAUSS_SCALE = (0.02, 0.08)

# Rozmycie
_ROZMYCIE_RUCH_I_SREDNICA = (3, 7)
_ROZMYCIE_GAUSS_SIGMA = (0.3, 1.2)

# Kompresja JPEG (zapis/odczyt w pamięci)
_JPEG_JAKOSC_MIN = 60
_JPEG_JAKOSC_MAX = 95

# Jasność i kontrast (znormalizowane: -1.0..1.0)
_JASNOSC_KONTRAST_JAS = (-0.10, 0.10)
_JASNOSC_KONTRAST_KONTR = (-0.15, 0.15)

# Nasycenie i odcień
_NASYCENIE_HUE_SAT = (-15, 15)
_NASYCENIE_VAL_SAT = (-20, 20)

# Winietowanie
_WINIETA_INTENS_MIN = 0.15
_WINIETA_INTENS_MAX = 0.40

# Chroma aberration
_CHROMA_PRzesuniecie_MAX = 3
_CHROMA_ROZMYCIE = 0.6

# Ekspozycja lokalna
_LOKALNA_ROZMIAR_OKNA_MIN = 80
_LOKALNA_ROZMIAR_OKNA_MAX = 250
_LOKALNA_ROZMIAR_SIGMA = 0.25
_LOKALNA_WARIACJA = 0.12

# Balans bieli (temperatura barwowa)
_BBT_MIN = 5500.0
_BBT_MAX = 7500.0
_BBT_ODCHYLENIE = 1000.0   # 100 K -> 0.5% wzmocnienia kanału (max ±5%)


# =============================================================================
# Funkcje pomocnicze (czyste transformacje pikselowe)
# =============================================================================

def _kompresja_jpeg(obraz: NDArray, jakosc: int) -> NDArray:
    """Kompresja JPEG w pamięci — symulacja artefaktów aparatu."""
    param = [int(cv2.IMWRITE_JPEG_QUALITY), jakosc]
    bufor, bufor_odwrotny = cv2.imencode(".jpg", obraz, param)
    return cv2.imdecode(bufor_odwrotny, cv2.IMREAD_COLOR)


def _winietowanie(obraz: NDArray, intensywnosc: float) -> NDArray:
    """Ciemnienie krawędzi i rogów kadru — efekt winiety obiektywu.

    Profil zbliżony do cos^4: siła rośnie z kwadratem odległości od
    środka, pełna intensywność dopiero w narożnikach kadru.
    """
    wys, szer = obraz.shape[:2]
    ox, oy = szer / 2.0, wys / 2.0
    y, x = np.mgrid[0:wys, 0:szer]
    odleglosc = np.sqrt(((x - ox) / ox) ** 2 + ((y - oy) / oy) ** 2)
    d_norm = odleglosc / 1.41421          # narożniki kadru -> 1.0
    maska = 1.0 - intensywnosc * d_norm ** 2
    return np.clip(obraz * maska[:, :, np.newaxis], 0, 255).astype(np.uint8)


def _aberracja_chromatyczna(obraz: NDArray, przesuniecie: int,
                            rozmycie: float) -> NDArray:
    """Delikatne rozchodzenie kanałów R/B — aberracja obiektywu.

    Siła przesunięcia rośnie z kwadratem odległości od środka kadru
    (radialnie, jak w prawdziwej optyce), zero na środku.

    UWAGA: obraz jest w kolejności BGR (cv2.imread), więc kanał czerwony
    to indeks 2, a niebieski to indeks 0 — nie odwrotnie.
    """
    wynik = obraz.copy()
    wys, szer = obraz.shape[:2]

    x, y = np.meshgrid(np.linspace(-1, 1, szer), np.linspace(-1, 1, wys))
    dx = x * (x * x + y * y) * przesuniecie / szer
    dx = dx.astype(np.float32)

    map_x_plus = (x + dx).astype(np.float32)
    map_x_minus = (x - dx).astype(np.float32)
    map_y = y.astype(np.float32)

    map_x_plus = (map_x_plus + 1) * 0.5 * szer
    map_x_minus = (map_x_minus + 1) * 0.5 * szer
    map_y_out = (map_y + 1) * 0.5 * wys

    # POPRAWKA: przeciwstawne przesunięcie kanału niebieskiego (0)
    # i czerwonego (2) zamiast modyfikacji tylko jednego kanału — to daje
    # realistyczny efekt
    # rozchodzenia barw zamiast jednostronnego przesunięcia.
    wynik[:, :, 0] = cv2.remap(
        cv2.GaussianBlur(obraz[:, :, 0], (0, 0), rozmycie),
        map_x_minus, map_y_out, cv2.INTER_LINEAR,
    )
    wynik[:, :, 2] = cv2.remap(
        cv2.GaussianBlur(obraz[:, :, 2], (0, 0), rozmycie),
        map_x_plus, map_y_out, cv2.INTER_LINEAR,
    )
    return wynik


def _balans_bieli(obraz: NDArray, temperatura: float) -> NDArray:
    """Zmiana temperatury barwowej: >6500K cieplej, <6500K chłodniej.

    UWAGA: obraz jest w kolejności BGR — kanał 0 to niebieski,
    kanał 2 to czerwony.
    """
    wynik = obraz.astype(np.float32)
    przesuniecie = (temperatura - 6500.0) / _BBT_ODCHYLENIE

    if przesuniecie > 0:
        wynik[:, :, 2] *= 1.0 + przesuniecie * 0.05   # czerwony w górę
        wynik[:, :, 0] *= 1.0 - przesuniecie * 0.03   # niebieski w dół
    else:
        wynik[:, :, 2] *= 1.0 + przesuniecie * 0.03   # czerwony w dół
        wynik[:, :, 0] *= 1.0 - przesuniecie * 0.05   # niebieski w górę

    return np.clip(wynik, 0, 255).astype(np.uint8)


def _ekspozycja_lokalna(obraz: NDArray) -> NDArray:
    """Subtelne zmiany ekspozycji w losowych obszarach kadru."""
    wynik = obraz.astype(np.float32)
    wys, szer = wynik.shape[:2]
    # POPRAWKA: użycie `random` zamiast `np.random`, żeby losowość była spójna
    # z resztą modułu i deterministyczna względem jednego globalnego seeda.
    rozmiar_okna = random.randint(
        _LOKALNA_ROZMIAR_OKNA_MIN, _LOKALNA_ROZMIAR_OKNA_MAX
    )

    maska_gauss = np.zeros((wys, szer), dtype=np.float32)
    cx = random.randint(rozmiar_okna, szer - rozmiar_okna)
    cy = random.randint(rozmiar_okna, wys - rozmiar_okna)
    rozmiar_sigma = rozmiar_okna * _LOKALNA_ROZMIAR_SIGMA

    maska_gauss[cy, cx] = 1.0
    maska_gauss = cv2.GaussianBlur(
        maska_gauss, (0, 0), rozmiar_sigma
    )
    maska_gauss /= maska_gauss.max() + 1e-7

    modyfikacja = random.uniform(
        1.0 - _LOKALNA_WARIACJA, 1.0 + _LOKALNA_WARIACJA
    )
    wynik *= 1.0 + maska_gauss[:, :, np.newaxis] * (modyfikacja - 1.0)
    return np.clip(wynik, 0, 255).astype(np.uint8)


# =============================================================================
# Główna funkcja augmentacji
# =============================================================================

def augmentuj_obraz(sciezka_wejsciowa: str, sciezka_wyjsciowa: str,
                    seed: int | None = None) -> None:
    """Aplikuje realistyczne efekty fotograficzne na obrazie PNG.

    Transformacje wyłącznie pikselowe — geometria obrazu i plik etykiet
    YOLO pozostają bez zmian. Kolejność efektów odpowiada fizycznej
    akwizycji: optyka/ruch -> szum sensora -> obróbka w aparacie (ISP).

    seed — opcjonalny; daje pełną odtwarzalność efektu dla tego obrazu.
    """
    obraz = cv2.imread(sciezka_wejsciowa, cv2.IMREAD_COLOR)
    if obraz is None:
        raise FileNotFoundError(
            f"Nie udało się wczytać obrazu: {sciezka_wejsciowa}"
        )

    if seed is not None:
        random.seed(seed)

    # parametry własnych efektów losowane raz na obraz (z globalnego random)
    temperatura = random.uniform(_BBT_MIN, _BBT_MAX)
    winieta = random.uniform(_WINIETA_INTENS_MIN, _WINIETA_INTENS_MAX)
    chroma_dx = random.randint(1, _CHROMA_PRzesuniecie_MAX)
    jpeg_q = random.randint(_JPEG_JAKOSC_MIN, _JPEG_JAKOSC_MAX)

    pipeline = A.Compose(
        [
            # -- optyka i ruch kamery --
            A.OneOf([
                A.MotionBlur(blur_limit=_ROZMYCIE_RUCH_I_SREDNICA, p=1.0),
                A.GaussianBlur(blur_limit=_ROZMYCIE_RUCH_I_SREDNICA,
                               sigma_limit=_ROZMYCIE_GAUSS_SIGMA, p=1.0),
            ], p=0.35),
            # -- szum sensora --
            A.OneOf([
                A.GaussNoise(std_range=_SZUM_GAUSS_SCALE, p=1.0),
                A.ISONoise(intensity=(0.1, 0.5), color_shift=(0.01, 0.05),
                           p=1.0),
            ], p=0.45),
            # -- obróbka w aparacie (ISP) --
            A.RandomBrightnessContrast(
                brightness_limit=_JASNOSC_KONTRAST_JAS,
                contrast_limit=_JASNOSC_KONTRAST_KONTR,
                brightness_by_max=True, p=0.50,
            ),
            A.HueSaturationValue(
                hue_shift_limit=_NASYCENIE_HUE_SAT,
                sat_shift_limit=_NASYCENIE_VAL_SAT,
                val_shift_limit=_NASYCENIE_VAL_SAT, p=0.45,
            ),
            A.Lambda(image=lambda im, **kw: _balans_bieli(im, temperatura),
                     p=0.40),
            A.Lambda(image=lambda im, **kw: _winietowanie(im, winieta),
                     p=0.45),
            A.Lambda(image=lambda im, **kw: _aberracja_chromatyczna(
                im, chroma_dx, _CHROMA_ROZMYCIE), p=0.25),
            A.Lambda(image=lambda im, **kw: _ekspozycja_lokalna(im), p=0.30),
            A.Lambda(image=lambda im, **kw: _kompresja_jpeg(im, jpeg_q),
                     p=0.40),
        ],
        seed=seed,
    )

    wynik = pipeline(image=obraz)["image"]
    cv2.imwrite(sciezka_wyjsciowa, wynik)


# =============================================================================
# Batch: augmentacja całego splitu train (kopie _augK + duplikaty etykiet)
# =============================================================================

def augmentuj_dataset(katalog_datasetu: str, kopie: int = 1,
                      seed: int | None = None) -> None:
    """Tworzy kopie augmented (_augK) każdego obrazu ze splitu train.

    Oryginalne pliki pozostają nietknięte; labelka .txt kopiowana 1:1
    (transformacje są wyłącznie pikselowe). Rerun jest bezpieczny —
    pliki *_aug* nie są ponownie augmentowane. Przy podanym seedzie
    każdy obraz dostaje własny, pochodny i stabilny seed (crc32 nazwy),
    więc efekt jest różny między obrazami, ale odtwarzalny w rerun.
    """
    if seed is not None:
        random.seed(seed)

    kat_img = os.path.join(katalog_datasetu, "images", "train")
    kat_lbl = os.path.join(katalog_datasetu, "labels", "train")
    if not os.path.isdir(kat_img):
        raise FileNotFoundError(f"Brak katalogu: {kat_img}")

    zrodla = sorted(
        p for p in glob.glob(os.path.join(kat_img, "*.png"))
        if "_aug" not in os.path.basename(p)
    )
    print(f"Obrazy źródłowe w train: {len(zrodla)}, kopii na obraz: {kopie}")

    zrobione = 0
    for sciezka in zrodla:
        baza = os.path.splitext(os.path.basename(sciezka))[0]
        labelka = os.path.join(kat_lbl, baza + ".txt")
        if not os.path.isfile(labelka):
            print(f"POMIJAM (brak labelki): {baza}")
            continue

        for k in range(1, kopie + 1):
            out_img = os.path.join(kat_img, f"{baza}_aug{k}.png")
            out_lbl = os.path.join(kat_lbl, f"{baza}_aug{k}.txt")
            seed_img = (None if seed is None else
                        (zlib.crc32(baza.encode("utf-8")) ^ seed) % 2**31)
            augmentuj_obraz(sciezka, out_img, seed_img)
            shutil.copyfile(labelka, out_lbl)

        zrobione += 1
        if zrobione % 100 == 0:
            print(f"[{zrobione}/{len(zrodla)}] zaugmentowane")

    print(f"Gotowe! Augmentacji poddano {zrobione} obrazów "
          f"(+{zrobione * kopie} kopii _aug).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augmentacja pikselowa splitu train datasetu YOLO.")
    parser.add_argument("dataset",
                        help="katalog datasetu, np. data/lego_dataset")
    parser.add_argument("--kopie", type=int, default=1,
                        help="liczba kopii augmented na obraz (default: 1)")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed dla pełnej odtwarzalności")
    args = parser.parse_args()
    augmentuj_dataset(args.dataset, args.kopie, args.seed)


if __name__ == "__main__":
    main()
