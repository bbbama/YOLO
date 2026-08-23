"""
Moduł augmentacji obrazów dla datasetu YOLO.
Augmentacja wyłącznie pikselowa — geometria i etykiety bez zmian.

Instalacja:
    pip install albumentations opencv-python pillow numpy
"""

import random

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
_ROZMYCIE_GAUSS_SIGMA     = (0.3, 1.2)

# Kompresja JPEG (zapis/odczyt w pamięci)
_JPEG_JAKOSC_MIN = 60
_JPEG_JAKOSC_MAX = 95

# Jasność i kontrast (znormalizowane: -1.0..1.0)
_JASNOSC_KONTRAST_JAS   = (-0.10, 0.10)
_JASNOSC_KONTRAST_KONTR = (-0.15, 0.15)

# Nasycenie i odcień
_NASYCENIE_HUE_SAT = (-15, 15)
_NASYCENIE_VAL_SAT = (-20, 20)

# Winietowanie
_WINIETA_PROMIEN_MIN = 0.4
_WINIETA_PROMIEN_MAX = 0.85
_WINIETA_INTENS_MIN  = 0.3
_WINIETA_INTENS_MAX  = 0.7

# Chroma aberration
_CHROMA_PRzesuniecie_MAX = 3
_CHROMA_ROZMYCIE         = 0.6

# Ekspozycja lokalna
_LOKALNA_ROZMIAR_OKNA_MIN = 80
_LOKALNA_ROZMIAR_OKNA_MAX = 250
_LOKALNA_ROZMIAR_SIGMA    = 0.25
_LOKALNA_WARIACJA         = 0.12

# Balans bieli (temperatura barwowa)
_BBT_MIN = 5500.0
_BBT_MAX = 7500.0
_BBT_ODCHYLENIE = 100.0


# =============================================================================
# Funkcje pomocnicze (czyste transformacje pikselowe)
# =============================================================================

def _kompresja_jpeg(obraz: NDArray, jakosc: int) -> NDArray:
    """Kompresja JPEG w pamięci — symulacja artefaktów aparatu."""
    param = [int(cv2.IMWRITE_JPEG_QUALITY), jakosc]
    bufor, bufor_odwrotny = cv2.imencode(".jpg", obraz, param)
    return cv2.imdecode(bufor_odwrotny, cv2.IMREAD_COLOR)


def _winietowanie(obraz: NDArray, promien: float, intensywnosc: float) -> NDArray:
    """Ciemnienie krawędzi kadru — efekt winiety obiektywu."""
    wys, szer = obraz.shape[:2]
    ox, oy = szer / 2.0, wys / 2.0
    # POPRAWKA: kolejność osi w mgrid musi odpowiadać (wys, szer) z obraz.shape,
    # inaczej przy niekwadratowych obrazach winieta wychodzi transponowana.
    y, x = np.mgrid[0:wys, 0:szer]
    odleglosc = np.sqrt(((x - ox) / ox) ** 2 + ((y - oy) / oy) ** 2)
    maska = np.clip(odleglosc / promien, 0.0, 1.0)
    maska = 1.0 - maska * intensywnosc
    return np.clip(obraz * maska[:, :, np.newaxis], 0, 255).astype(np.uint8)


def _aberracja_chromatyczna(obraz: NDArray, przesuniecie: int,
                            rozmycie: float) -> NDArray:
    """Delikatne rozchodzenie kanałów R/B — aberracja obiektywu.

    UWAGA: obraz jest w kolejności BGR (cv2.imread), więc kanał czerwony
    to indeks 2, a niebieski to indeks 0 — nie odwrotnie.
    """
    wynik = obraz.copy()
    wys, szer = obraz.shape[:2]

    x, y = np.meshgrid(np.linspace(-1, 1, szer), np.linspace(-1, 1, wys))
    dx = (x + y) * przesuniecie / szer
    dx = dx.astype(np.float32)

    map_x_plus = (x + dx).astype(np.float32)
    map_x_minus = (x - dx).astype(np.float32)
    map_y = y.astype(np.float32)

    map_x_plus = (map_x_plus + 1) * 0.5 * szer
    map_x_minus = (map_x_minus + 1) * 0.5 * szer
    map_y_out = (map_y + 1) * 0.5 * wys

    # POPRAWKA: przeciwstawne przesunięcie kanału niebieskiego (0) i czerwonego (2)
    # zamiast modyfikacji tylko jednego kanału — to daje realistyczny efekt
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

    UWAGA: obraz jest w kolejności BGR — kanał 0 to niebieski, kanał 2 to czerwony.
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

def augmentuj_obraz(sciezka_wejsciowa: str, sciezka_wyjsciowa: str) -> None:
    """Losowo aplicuje realistyczne efekty fotograficzne na obrazie PNG.

    Transformacje wyłącznie pikselowe — geometria obrazu i plik etykiet
    YOLO pozostają bez zmian.
    """
    obraz = cv2.imread(sciezka_wejsciowa, cv2.IMREAD_COLOR)
    if obraz is None:
        raise FileNotFoundError(
            f"Nie udało się wczytać obrazu: {sciezka_wejsciowa}"
        )

    # -- szum sensora (Gauss lub ISO) --
    if random.random() < 0.45:
        szum_typ = random.choice(["gauss", "iso"])
        if szum_typ == "gauss":
            obraz = A.GaussNoise(
                p=1.0,
                std_range=_SZUM_GAUSS_SCALE,
            )(image=obraz)["image"]
        else:
            obraz = A.ISONoise(
                p=1.0,
                intensity=(0.1, 0.5),
                color_shift=(0.01, 0.05),
            )(image=obraz)["image"]

    # -- rozmycie (ruch lub gauss) --
    if random.random() < 0.35:
        rozmycie_typ = random.choice(["ruch", "gauss"])
        if rozmycie_typ == "ruch":
            obraz = A.MotionBlur(
                p=1.0,
                blur_limit=_ROZMYCIE_RUCH_I_SREDNICA,
            )(image=obraz)["image"]
        else:
            obraz = A.GaussianBlur(
                p=1.0,
                blur_limit=_ROZMYCIE_RUCH_I_SREDNICA,
                sigma_limit=_ROZMYCIE_GAUSS_SIGMA,
            )(image=obraz)["image"]

    # -- kompresja JPEG --
    if random.random() < 0.40:
        jakosc = random.randint(_JPEG_JAKOSC_MIN, _JPEG_JAKOSC_MAX)
        obraz = _kompresja_jpeg(obraz, jakosc)

    # -- jasność i kontrast --
    if random.random() < 0.50:
        obraz = A.RandomBrightnessContrast(
            p=1.0,
            brightness_limit=_JASNOSC_KONTRAST_JAS,
            contrast_limit=_JASNOSC_KONTRAST_KONTR,
            brightness_by_max=True,
        )(image=obraz)["image"]

    # -- nasycenie i odcień --
    if random.random() < 0.45:
        obraz = A.HueSaturationValue(
            p=1.0,
            hue_shift_limit=_NASYCENIE_HUE_SAT,
            sat_shift_limit=_NASYCENIE_VAL_SAT,
            val_shift_limit=_NASYCENIE_VAL_SAT,
        )(image=obraz)["image"]

    # -- balans bieli (temperatura barwowa) --
    if random.random() < 0.40:
        temperatura = random.uniform(_BBT_MIN, _BBT_MAX)
        obraz = _balans_bieli(obraz, temperatura)

    # -- winietowanie --
    if random.random() < 0.45:
        promien = random.uniform(_WINIETA_PROMIEN_MIN, _WINIETA_PROMIEN_MAX)
        intensywnosc = random.uniform(_WINIETA_INTENS_MIN, _WINIETA_INTENS_MAX)
        obraz = _winietowanie(obraz, promien, intensywnosc)

    # -- aberracja chromatyczna --
    if random.random() < 0.25:
        przesuniecie = random.randint(1, _CHROMA_PRzesuniecie_MAX)
        obraz = _aberracja_chromatyczna(obraz, przesuniecie, _CHROMA_ROZMYCIE)

    # -- ekspozycja lokalna --
    if random.random() < 0.30:
        obraz = _ekspozycja_lokalna(obraz)

    # -- zapis wyniku --
    cv2.imwrite(sciezka_wyjsciowa, obraz)