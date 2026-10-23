# Detekcja i liczenie klocków LEGO przy użyciu YOLO oraz Blendera




## Detekcja i liczenie klocków LEGO przy użyciu sieci YOLO i syntetycznego zbioru danych z Blendera

*Wykorzystanie generowania grafiki 3D do automatycznego uczenia maszynowego w zadaniach wizji komputerowej*

* **Prelegent:** Bartosz Amalio
* **Projekt:** Inteligentne liczenie obiektów w oparciu o głębokie sieci neuronowe
* **Główne technologie:** Blender 3D (Python API), Ultralytics YOLO11, akceleracja GPU.

---

## Cel projektu i problem badawczy
### Jak efektywnie uczyć sieci neuronowe bez czasochłonnego etykietowania?

* **Wyzwanie w Computer Vision:** Modele takie jak YOLO wymagają tysięcy różnorodnych zdjęć do poprawnej nauki. Ręczne rysowanie obwiedni (bounding boxów) zajmuje setki godzin i jest podatne na błędy ludzkie.
* **Rozwiązanie:** Zastąpienie rzeczywistych zdjęć w pełni automatycznym procesem generowania fotorealistycznych scen 3D w programie Blender.
* **Cel praktyczny:** Stworzenie systemu liczącego klocki LEGO typu `lego_2x4` w czasie rzeczywistym, odpornego na zmiany oświetlenia, tła oraz ułożenia obiektów.

---

## Tworzenie datasetu - Dlaczego Blender?
### Zalety generowania syntetycznego datasetu 3D

* **Skalowalność bez kosztów:** Wygenerowanie 1000+ unikalnych obrazów treningowych w kilka minut za pomocą skryptu [Simulation.py]
* **Automatyczne i bezbłędne etykietowanie:** Pozycje bounding boxów są obliczane matematycznie bezpośrednio z rzutu kamery 3D na płaszczyznę dwuwymiarową.
* **Idealny podział danych:** Automatyczny podział zbioru na części:
  * **Train** (70%) - do właściwej nauki modelu
  * **Val** (15%) - do weryfikacji w trakcie treningu
  * **Test** (15%) - do ostatecznego sprawdzenia skuteczności.

---

## Algorytm generowania sceny
### Jak skrypt układa klocki w wirtualnym świecie?

* **Losowa liczba obiektów:** W każdej iteracji skrypt generuje od 1 do 10 klocków na scenie.
* **Modyfikacja orientacji 3D:** Klocki mogą leżeć stabilnie na płasko, do góry nogami, lub na bokach (dłuższych/krótszych), co uczy model rozpoznawania obiektów pod różnymi kątami.
* **Zapobieganie przenikaniu (kolizjom):** Użycie algorytmu okręgów opisanych na płaszczyźnie XY. Skrypt szuka wolnego miejsca, a następnie precyzyjnie obniża klocek w osi Z tak, aby leżał idealnie na powierzchni półki.
* **Automatyczne czyszczenie:** Po wyrenderowaniu klatki, wygenerowane klocki są usuwane z pamięci Blendera, zapobiegając wyciekom zasobów.

---

## Randomizacja domeny (Domain Randomization)
### Przygotowanie modelu na warunki świata rzeczywistego

Aby model wyszkolony w Blenderze działał na zwykłych zdjęciach, środowisko 3D jest maksymalnie różnicowane:
* **Losowe oświetlenie:** Od 1 do 3 źródeł światła różnych typów (Point, Sun, Area), o zmiennej mocy (200-1500W), barwie oraz pozycji.
* **Zmienne tekstury i materiały:** 
  * Losowanie fizycznych tekstur półki oraz obrazów tła (HDRI).
  * Losowy kolor klocków oraz parametry chropowatości (Roughness) i metaliczności (Metallic).
* **Parametry optyczne kamery:** Losowa pozycja kamery na sferze, zmienna ogniskowa (20-70 mm) oraz głębia ostrości (Depth of Field) nakładana na losowo wybrany klocek.

---

## Obiekty zakłócające (Dystraktory) i Edge Cases
### Budowanie odporności na zakłócenia i sytuacje skrajne

* **Dystraktory:** Losowo generowane bryły (kule, sześciany, torusy, stożki) omijające klocki. Nie są one etykietowane, dzięki czemu model uczy się je ignorować i nie mylić ich z klockami LEGO.
* **Sytuacje skrajne (Edge Cases):**
  * **Zasłonięcia (Occlusions):** Klocki mogą się częściowo zasłaniać.
  * **Krawędzie kadru:** Skrypt oblicza widoczność klocka. Jeśli jest widoczny w mniej niż 20% (np. wychodzi poza kadr), etykieta YOLO jest odrzucana.
  * **Rozmycie ruchu (Motion Blur):** Symulowanie niedoskonałości fizycznego aparatu poprzez losową wartość otwarcia migawki w Blenderze.

---

## Proces uczenia - Konfiguracja YOLO
### Trening modelu w skrypcie [train.py]

* **Architektura:** YOLO11n (wersja *nano*) - najlżejsza i najszybsza wersja sieci, idealna do wdrożeń na urządzeniach o ograniczonej mocy obliczeniowej.
* **Hiperparametry uczenia:**
  * **Liczba epok:** 50
  * **Rozmiar wejściowy obrazu (imgsz):** 416 x 416 pikseli
  * **Rozmiar paczki (batch size):** 32
  * **Akceleracja sprzętowa:** Trening na chmurowej maszynie serwerowej (**GPUHub**) z wykorzystaniem wydajnych układów graficznych **GPU**
* **Konfiguracja klas:** Plik `data.yaml` definiujący pojedynczą klasę wykrywania (`names: ['lego_2x4']`).

---

## Wyniki i stabilność treningu
### Jak uczył się nasz model?

* **Szybka zbieżność:** Model osiągnął pełną skuteczność klasyfikacji i lokalizacji niezwykle szybko - już około 10 epoki wskaźnik mAP@0.5 zbliżył się do 0.99.
* **Wartości strat w ostatniej (50.) epoce:**
  * Strata lokalizacji boxu (`train/box_loss`): **0.1271** (walidacja: **0.0964**)
  * Strata klasyfikacji (`train/cls_loss`): **0.1406** (walidacja: **0.1227**)
* **Brak przeuczenia (Overfittingu):** Straty walidacyjne były stabilne i utrzymywały tendencję spadkową, co dowodzi świetnej generalizacji wiedzy przez model.

---

## Ewaluacja i skuteczność modelu
### Wyniki uzyskane przez skrypt [evaluate_model.py]

Ewaluacja przeprowadzona na odseparowanym, niewidzianym wcześniej zbiorze testowym wykazała perfekcyjne rezultaty (przy progach `conf=0.70` oraz `IoU=0.50`):
* **mAP@0.5 (Mean Average Precision): 1.0000 (100%)**
  * Model z idealną precyzją lokalizuje obiekty w przestrzeni obrazu.
* **Zgodność liczby obiektów: 100.00%**
  * We wszystkich obrazach testowych model poprawnie określił dokładną liczbę klocków.
  * Pełna odporność na fałszywe wykrycia (False Positives) w obecności dystraktorów.

---

## Detekcja z kamery w czasie rzeczywistym
### Działanie systemu w praktyce (skrypt [camera_detect.py])

* **Płynna detekcja live:** Przetwarzanie obrazu w czasie rzeczywistym z kamery (np. wbudowanej lub USB) z wykorzystaniem wytrenowanego modelu `best.pt`.
* **Kluczowe parametry skryptu:**
  * **Źródło obrazu (`source="1"`):** Przechwytywanie strumienia wideo z wybranej kamery.
  * **Wyświetlanie wyników (`show=True`):** Podgląd wideo na żywo z nałożonymi obwiedniami detekcji (bounding box) oraz stopniem pewności.
  * **Próg ufności (`conf=0.4`):** Dobrany próg pewności wykrycia dla działania w warunkach rzeczywistych.
  * **Tryb strumienia (`stream=True`):** Przetwarzanie generatorowe (klatka po klatce) oszczędzające pamięć RAM i zapewniające wysoką płynność (FPS).

---

## Podsumowanie i dalsze kroki
### Co udało się osiągnąć i co dalej?

* **Podsumowanie:** Podejście syntetyczne oparte na Blenderze okazało się w pełni sukcesem. Model YOLO trenowany wyłącznie na wirtualnych renderach potrafi bezbłędnie liczyć klocki LEGO.
* **Perspektywy rozwoju projektu:**
  1. **Wieloklasowość:** Rozszerzenie generatora o inne typy klocków (np. 2x2, 1x4, 2x8) oraz naukę ich rozróżniania lub inne typy przedmiotów.
  2. **Optymalizacja brzegowa:** Przystosowanie modelu do działania mikrokomputerze (np. Raspberry Pi).


urealnienie obrazow klockow, poprzez zmiany w kolorach na dlugosci klocka

pozniej uzyc generatora obrazu

metrykki MOT

uzyc filtrów i rozmycia i szumu

zrobic zdjecie telefonem klocka i sprobowac je scharakteryzowac zeby pozniej na podastwie jego charakterystki tworzyc zdjecia