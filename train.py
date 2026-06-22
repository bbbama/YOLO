from ultralytics import YOLO
import os

def main():
    # 1. Załaduj model bazowy
    # Używamy wersji 'nano' (n), bo jest najszybsza do nauki i testów
    model = YOLO("yolo11n.pt")

    # 2. Ścieżka do Twoich danych
    data_path = "lego_dataset/data.yaml"

    print(f"Rozpoczynam trening na podstawie: {data_path}")

    # 3. Uruchom proces uczenia
    model.train(
        data=data_path,
        epochs=40,        # liczba powtórzeń (możesz zwiększyć, jeśli model słabo widzi)
        imgsz=416,         # rozmiar obrazu
        batch=32,          # ile zdjęć naraz (zmniejsz do 8 jeśli wywali błąd pamięci)
        cache=True,
        amp=True,
        device="mps",      # "mps" przyspiesza trening na Macu
        name="lego_nowy",  # nazwa folderu w runs/detect/
        exist_ok=True      # nadpisuj folder, jeśli już istnieje
    )

if __name__ == "__main__":
    main()
