# Dokumentacja techniczna projektu
## Syntetyczny dataset LEGO → detekcja i liczenie obiektów YOLO

> Materiał do przygotowania obrony pracy inżynierskiej.
> Konwencja: przy każdej funkcji omawiamy **co robi → jak działa krok po kroku → dlaczego tak**.
> To ostatnie jest najważniejsze — komisja prawie zawsze pyta „dlaczego właśnie tak?".

---

## Spis treści

1. [Wprowadzenie — o co chodzi w projekcie](#1-wprowadzenie)
2. [Struktura repozytorium](#2-struktura-repozytorium)
3. [Teoria minimum](#3-teoria-minimum)
4. [Generator scen — `src/simulation/Simulation.py`](#4-generator-scen)
5. [Augmentacja — `src/augmentuj.py`](#5-augmentacja)
6. [Trening — `src/detection/train.py`](#6-trening)
7. [Ewaluacja — `evaluate_model.py` / `model_ev.py`](#7-ewaluacja)
8. [Predykcja i detekcja na żywo](#8-predykcja-i-detekcja-na-żywo)
9. [Uruchomienie całego pipeline'u](#9-uruchomienie-całego-pipelineu)
10. [Pytania komisji + gotowe odpowiedzi (FAQ)](#10-pytania-komisji--gotowe-odpowiedzi-faq)

---

# 1. Wprowadzenie

## 1.1 Cel projektu

Celem jest **detekcja i liczenie klocków LEGO 2×4** na zdjęciach za pomocą sieci neuronowej YOLO.
Kluczowy pomysł pracy: **model jest trenowany wyłącznie (lub prawie wyłącznie) na danych syntetycznych** generowanych w Blenderze, a potem ma działać na prawdziwych zdjęciach. Ten „skok" nazywa się **transferem symulacja→rzeczywistość (sim-to-real)**.

Dlaczego to ma sens?
- Ręczne robienie i **etykietowanie** tysięcy zdjęć jest drogie i czasochłne.
- W Blenderze etykiety (bounding boxy) dostajemy **za darmo i idealnie** — wiemy dokładnie, gdzie stoi każdy klocek.
- Możemy wygenerować **dowolną liczbę** scen z dowolną różnorodnością.

Główne ryzyko: model nauczony na idealnych renderach może nie rozpoznać brzydkiego, rozmazanego zdjęcia z telefonu. Dlatego cała inżynieria tego projektu kręci się wokół jednego pytania: **jak zrobić syntetyki wystarczająco podobne do rzeczywistości?** Odpowiedzi są trzy warstwy:
1. **Realizm generatora** (materiały, światła, DOF, tekstury zużycia) — sekcja 4,
2. **Domain randomization** (losowość wszystkiego, żeby rzeczywistość była tylko „jedną z losowań") — sekcje 3.6 i 4,
3. **Augmentacja fotograficzna** (symulacja wad prawdziwego aparatu) — sekcja 5.

## 1.2 Pipeline w jednym obrazku

```
┌─────────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
│ Blender         │   │ Augmentacja  │   │ Trening YOLO │   │ Ewaluacja    │   │ Zastosowanie     │
│ Simulation.py   │──▶│ augmentuj.py │──▶│ train.py     │──▶│ evaluate_    │──▶│ predict.py       │
│ render + GT     │   │ efekty foto  │   │ transfer     │   │ model.py     │   │ camera_detect.py │
│ (data/lego_     │   │ (tylko train)│   │ learning     │   │ IoU/AP własne│   │ kamera live      │
│  dataset/)      │   └──────────────┘   └──────────────┘   └──────────────┘   │ liczenie klocków │
└─────────────────┘                                                             └──────────────────┘
```

- **GT (ground truth)** — „prawda bazowa", czyli idealne etykiety mówiące gdzie naprawdę są obiekty. W projekcie GT generuje się automatycznie razem z obrazem.
- Dataset ląduje w `data/lego_dataset/` w formacie YOLO (opisany w 3.2).

---

# 2. Struktura repozytorium

| Plik / katalog | Co zawiera |
|---|---|
| `src/simulation/Simulation.py` | Skrypt Pythona wykonywany **wewnątrz Blendersa** — generuje cały dataset (rendery + etykiety) |
| `src/simulation/Simulation.blend` | Scena bazowa: wzorzec klocka LEGO 2×4, półka, kamera |
| `src/augmentuj.py` | Augmentacja fotograficzna splitu train (albumentations + OpenCV) |
| `src/detection/train.py` | Trening YOLO11n (ultralytics) |
| `src/detection/predict.py` | Jednorazowa predykcja na zdjęciu |
| `src/detection/camera_detect.py` | Detekcja na żywo z kamery |
| `src/detection/evaluate_model.py` | Ewaluacja: własna implementacja IoU i AP |
| `src/detection/model_ev.py` | Duplikat ewaluacji (różni się tylko ścieżką do modelu: `train-2` vs `train2`) |
| `data/lego_dataset/` | Wygenerowany dataset: `images/{train,val,test}`, `labels/{train,val,test}`, `data.yaml`, `config.json` |
| `data/textures/backgrounds/` | Tekstury tła (HDRI/kolory) — obecnie puste, działa fallback |
| `data/textures/shelf/` | Tekstury nakładane na półkę |
| `models/` | Wagi bazowe (`yolo11n.pt`) i wytrenowane (`best.pt`) |
| `runs/detect/...` | Wyniki treningów ultralytics (wagi `best.pt`, wykresy) |

---

# 3. Teoria minimum

> Ta sekcja tłumaczy pojęcia, które warto umieć wytłumaczyć własnymi słowami. Bez wzorów z podręczników — tak, żeby zrozumieć i zapamiętać.

## 3.1 Jak działa YOLO?

YOLO (**You Only Look Once**) to **jednoetapowy detektor obiektów**. Stare podejścia robiły to dwuetapowo: najpierw zgadywały *gdzie* może być coś ciekawego (setki propozycji), a potem klasyfikowały każdy fragment. YOLO robi wszystko **w jednym przejściu** przez sieć — stąd nazwa.

Jak to możliwe? Obraz dzielony jest siatką (gridem) na komórki. Każda komórka „obiecuje", że w jej okolicy znajdują się obiekty, i dla każdego obiektu przewiduje:
- **bounding box** — prostokąt `(x, y, szerokość, wysokość)` otaczający obiekt,
- **confidence** (pewność) — liczba 0–1: „jak bardzo jestem pewien, że tu naprawdę jest obiekt",
- **klasę** — u nas jedna klasa: `lego_2x4`.

Sieć trenowana jest tak, by jej przewidywania były jak najbardziej zbliżone do etykiet GT. Używamy wersji **YOLO11n** („nano") z biblioteki **Ultralytics** — najmniejsza odmiana, szybka do treningu na laptopie, a przy jednej klasie i prostych klockach w zupełności wystarczająca.

## 3.2 Format etykiet YOLO

Każdy obraz `img_0001.png` ma plik `img_0001.txt`. Każdy wiersz = jeden obiekt:

```
class_id x_center y_center width height
0        0.512340 0.483210  0.180  0.240
```

Ważne detale (o to można zostać spytany):
- współrzędne są **znormalizowane do zakresu [0,1]** — dzielone przez szerokość/wysokość obrazu. Dzięki temu format jest niezależny od rozdzielczości,
- podawany jest **środek** prostokąta, nie lewy górny róg,
- **pusty plik `.txt` to poprawna etykieta** — mówi YOLO „na tym zdjęciu nic nie ma". Takie zdjęcia (tzw. **ramki tła**) uczą sieć redukować fałszywe alarmy. Nasz generator celowo produkuje ~10% takich scen.

## 3.3 IoU (Intersection over Union)

Podstawowa miara „jak dobrze prostokąt przewidziony pokrywa się z prawdziwym":

```
IoU = pole przecięcia boxów / pole sumy boxów
```

Dwa identyczne boxy mają IoU = 1, dwa nietkające się — IoU = 0. Umowny próg akceptacji dopasowania to często **IoU ≥ 0.5**: przecięcie pokrywa ponad połowę sumy powierzchni. Predykcja z IoU ≥ progu względem jakiegoś GT liczy się jako trafienie (**TP**, true positive); inaczej to fałszywy alarm (**FP**, false positive).

## 3.4 Precision, Recall, AP, mAP@0.5

- **Precision** = TP / (TP + FP): „z tego, co wykryłem, ile było naprawdę?"
- **Recall** = TP / (TP + FN): „z tego, co istnieje, ile wykryłem?"

Obie metryki zależą od progu confidence: obniżysz próg → więcej detekcji → recall rośnie, precision zwykle spada. Rysując precision w funkcji recall dla wszystkich progów dostajesz **krzywą PR**. Pole pod nią to **AP (Average Precision)** — jedna liczba podsumowująca jakość detektora dla jednej klasy.

**mAP@0.5** to średnia AP po wszystkich klasach przy progu dopasowania IoU = 0.5. U nas jedna klasa, więc mAP ≈ AP. Wartość 1.0 = perfekcja.

Dodatkowa, „biznesowa" metryka w projekcie: **zgodność liczenia** — na ilu % zdjęć liczba predykcji równa się liczbie obiektów GT. To bezpośrednia miara celu końcowego (liczenie klocków), dużo surowsza niż mAP: jedno zdublowane lub przeoczone pudełko psuje wynik zdjęcia.

## 3.5 Domain gap i domain randomization

**Domain gap** to „przepaść statystyczna" między domeną treningową (idealne rendery) a docelową (zdjęcia z telefonu): inne szumy, inne światła, inne materiały. Model widzący tylko rendery potrafi uczyć się **artefaktów renderera** zamiast cech klocka — wtedy w realu zawodzi.

**Domain randomization** to strategia walki z tym: **randomyzujemy wszystko, co nie jest istotą zadania**, aż rzeczywistość stanie się „jednym z kolejnych losowań". Jeśli materiały klocka, światła, pozycja kamery, ostrość i tło losują się w ogromnych zakresach, to prawdziwe zdjęcie mieści się w tej samej dystrybucji i model nie musi się „douczać" różnicy.

W projekcie randomizacja występuje na trzech poziomach:
1. **scena** — losowe materiały proceduralne, kolory, światła (typ/liczba/energia/pozycja), kamera (pozycja sferyczna, ogniskowa), DOF, tło, półka, rozpraszacze,
2. **augmentacja pikselowa** — symulacja wad aparatu (szum, winieta, aberracje…) nakładana po renderze,
3. **struktura sceny** — liczba klocków 1–10, sceny puste, klocki częściowo zasłonięte przez rozpraszacze.

## 3.6 Renderowanie: EEVEE vs Cycles, Principled BSDF, DOF

**EEVEE vs Cycles.** Cycles śledzi promienie świetlne (path tracing) — fotorealizm, ale wolno. EEVEE rasteryzuje obraz w czasie rzeczywistym z podglądaniem sceny — wielokrotnie szybszy. Wybrano EEVEE, bo przy generowaniu setek obrazów czas renderu decyduje, a dla zadania detekcji (nie fotorealizmu!) jakość wystarcza. Jakość antyaliasingu regulujemy parametrem `taa_render_samples = 128`.

**Principled BSDF** to uniwersalny, fizycznie uzasadniony shader w Blenderze (tzw. workflow metallic-roughness). Najważniejsze wejścia, używane w kodzie:
- **Base Color** — bazowy kolor powierzchni,
- **Roughness** — 0 = lustro, 1 = idealnie matowa; to tu „wpisują się" zarysowania i zużycie,
- **Metallic** — metal vs dielektryk (plastik ≈ 0),
- **Coat Weight** — dodatkowy cienki lakierowany połysk na wierzchu (efekt błyszczącej plastikowej „szkliwa"),
- **Normal** — mikro-nierówności powierzchni (podkradamy je z węzła Bump).

**Fizyczny DOF (głębia ostrości).** Prawdziwy aparat ostry jest tylko w jednym płaszczyźnie; stopień rozmycia zależy od otworu przysłony. W optyce: **f-stop = ogniskowa / średnica otworu** (np. obiektyw 50 mm z otworem 25 mm → f/2). Mniejsze f-stop → większy otwór → większe rozmycie tła. Kod implementuje dokładnie ten wzór (sekcja 4.14), zamiast losować f-stop „z kapelusza".

---

# 4. Generator scen

Plik: `src/simulation/Simulation.py` (~700 linii). Uruchamiany **wewnątrz Blendersa** w trybie headless (bez okna):

```bash
blender -b src/simulation/Simulation.blend -P src/simulation/Simulation.py
```

Flaga `-b` = background (bez GUI), `-P` = wykonaj skrypt. Skrypt działa na otwartej scence `.blend`, w której czekają trzy obiekty: wzorzec klocka (`LEGO-2X4-L`), półka (`Cube2`) i `Camera`.

Logika skryptu: **NUM_SAMPLES razy** zbuduj losową scenę → ustaw kamerę i światła → wyrenderuj PNG → policz bounding boxy → zapisz parę PNG + TXT → posprzątaj.

## 4.1 Konfiguracja (sekcja KONFIGURACJA)

| Stała | Wartość | Znaczenie i dlaczego |
|---|---|---|
| `CLASS_ID` | 0 | Identyfikator jedynej klasy w etykietach (`lego_2x4`) |
| `NUM_SAMPLES` | 10 | Liczba renderów na jedno uruchomienie. Do produkcji podnosi się np. do 1000+ |
| `MIN_KLOCKI / MAX_KLOCKI` | 1 / 10 | Zakres losowania liczby klocków w scenie |
| `MAX_DYSTR` | 8 | Maksymalna liczba rozpraszaczy (distractors — patrz 4.13) |
| `TRAIN_RATIO / VAL_RATIO` | 0.70 / 0.15 | Proporcje splitów (reszta ≈ 15% idzie do test) |
| `NUM_LIGHTS_MIN/MAX` | 1 / 3 | Zakres liczby lamp w scenie |
| `P_BRAK_DOF` | 0.25 | 25% zdjęć renderowanych w pełni ostrych (bez DOF) — model ma widzieć też idealnie ostre ujęcia |
| `P_PUSTA_SCENA` | 0.10 | 10% scen bez klocków — ramki tła redukujące fałszywe alarmy |
| `SEED` | None | Wpisz liczbę → `random.seed()` → pełna odtwarzalność układu scen (ważne przy pisaniu pracy: recenzent może odtworzyć dataset) |
| `KRAWEDZ_POLKI` | 0.3 | Margines od fizycznej krawędzi półki, żeby klocek nie wystawał krawędzią |
| `ZAKRES_XY` | 2.0 | Limit odległości od środka półki. **Ważny kompromis:** półka w scenie ma ~60×18 j.u., więc pełny zakres dawałby klocki niewidoczne w kadrze i masowe odrzuty przez kolizje. ±2 j.u. trzyma obiekty blisko środka |
| `USE_TEX_BG / USE_TEX_SHELF` | True | Przełączniki tekstur tła i półki (z fallbackiem, gdy katalogi puste) |

Ścieżki budowane są względem lokalizacji pliku (`PROJECT_ROOT`), więc skrót działa z dowolnego katalogu roboczego. Na końcu bloku konfiguracji: jeśli `SEED` ustawione, wołane jest `random.seed(SEED)` — **wszystkie** losowania w skrypcie korzystają z modułu `random`, więc całość staje się deterministyczna.

## 4.2 Przygotowanie sceny i Bevel modifier

```python
obj_wzor  = bpy.data.objects["LEGO-2X4-L"]
obj_polka = bpy.data.objects["Cube2"]
cam       = bpy.data.objects["Camera"]
obj_wzor.hide_render = True; obj_wzor.hide_viewport = True
```

- **Wzorzec klocka** służy wyłącznie jako „matryca" do klonowania — jest ukrywany przed renderem, bo w scenie mają być tylko jego klony.
- Od razu do wzorca dodawany jest **Bevel modifier**: `width = max(dimensions) * uniform(0.005, 0.015)`, `segments = 2`. Zaokrągla on idealnie ostre krawędzie siatki. **Dlaczego:** prawdziwy klocek ma fazowane krawędzie, które łapią refleksy światła; idealna krawędź matematyczna wygląda sztucznie (CGI-look) i psuje też obliczenia `Pointiness` (patrz 4.5, sekcja 4 materiału). Modifier dodany do wzorca **dziedziczy się na wszystkie klony** — jeden wpis obsługuje całą scenę.

## 4.3 Renderer i struktura datasetu

Renderer: `'BLENDER_EEVEE'`, rozdzielczość **640×640** (kwadrat — brak zniekształceń proporcji przy skalowaniu w YOLO), `filter_size = 1.5` (filtr pikselowy antyaliasingu przy zapisie — niższy = ostrzejszy obraz; domyślne 2.0 lekko „rozmywało"), `taa_render_samples = 128` (128 próbek temporal anti-aliasing = gładkie krawędzie bez migotania). Zapis: PNG, RGB, kompresja 0 (szybciej; PNG i tak bezstratny).

Następnie tworzone są katalogi `images/{train,val,test}` i `labels/{train,val,test}` oraz plik **`data.yaml`** — „wizytówka" datasetu dla Ultralytics:

```yaml
path: .
train: images/train
val:   images/val
nc: 1              # number of classes
names: ['lego_2x4']
```

Split **test** nie jest wpisany do yaml — Ultralytics podczas walidacji używa `val`. Folder test służy naszej własnej ewaluacji (sekcja 7).

## 4.4 Funkcje pomocnicze: `wczytaj_tekstury`, `bsdf`, `nowy_material`

- **`wczytaj_tekstury(katalog)`** — zbiera globami wszystkie pliki graficzne (png/jpg/jpeg/bmp/tga/hdr/exr, także WERSZAMI LITER — Windows). Zwraca listę ścieżek; pusta lista = później zadziała fallback.
- **`bsdf(obj)`** — bezpiecznie wyciąga węzeł `Principled BSDF` z aktywnego materiału obiektu (lub `None`). Wszędzie indziej operujemy na wejściach tego węzła (`Base Color`, `Roughness`…), więc helper centralizuje obsługę braku materiału.
- **`nowy_material(obj, nazwa)`** — tworzy świeży materiał Blendera, przypina go do obiektu i zwraca węzeł BSDF. **Dlaczego świeży dla każdego klocka:** materiały w Blenderze są współdzielone przez referencję — gdyby klony dzieliły materiał wzorca, wszyscy mielibyśmy ten sam kolor. Każdy klocek dostaje własną kopię drzewa node'ów.

## 4.5 `losuj_material_klocka(obj, idx)` — proceduralna tekstura zużycia

Najobszerniejsza funkcja generatora. Buduje z node'ów shaderowych unikalny materiał klocka. Filozofia: **klocek nie może wyglądać jak jednolita plastikowa kostka** — model miałby uczyć się koloru zamiast kształtu. Dlatego materiał losuje spektrum zużycia.

Kluczowa zmienna: **`zuzywanie ∈ [0,1]`** (uniform) — 0 = klocek jak prosto z pudełka (gładki, błyszczący), 1 = mocno używany (matowy, porysowany, zabrudzony). Skaluje ona siłę wszystkich warstw naraz, więc dataset zawiera naturalne przejście między stanami klocków.

Losowane wartości bazowe: `kolor` — RGB z pełnego zakresu, `bazowy_metallic ∈ [0, 0.1]` (plastik, więc blisko zera).

Drzewo node'ów, warstwa po warstwie (numery jak w komentarzach kodu):

**(1) Mikro-zarysowania → Roughness.** Duży `Noise Texture` (scale 60–120, detail 8) → `ColorRamp`. Rampa mapuje szum na wartości chropowatości; przesunięcie jej progów zależy od zużycia (`0.35 − z·0.1` oraz `0.65 + z·0.1`), więc zużyte klocki mają szerszy, mocniejszy kontrast roughness.

**(2) Bump.** Drugi noise (scale 80–150) → węzeł `Bump` podpięty do wejścia **Normal**. Symuluje mikronierówności powierzchni bez zmiany geometrii. Siła: `0.02 + zuzywanie·[0.08–0.23]` — nowe klocki prawie gładkie, zużyte wyraźnie „chropowate".

**(3) Plamy koloru.** Wolnozmienny noise (scale 2–8 — duże plamy) → `MixRGB`: `Color1` = kolor bazowy, `Color2` = wyjście szumu, `Fac = min([0.05–0.15] + z·[0.10–0.25], 0.45)`. Efekt: delikatne przebarwienia typu wyblakły plastik, silniejsze wraz z zużyciem (ale nigdy ponad 45% mieszanki — kolor bazowy zawsze dominuje).

**(4) Ślady użytkowania na krawędziach (edge wear).** Węzeł `Geometry` udostępnia **`Pointiness`** — wartość 0..1 mówiącą, czy fragment powierzchni jest wypukły (krawędzie ≈ >0.5) czy wklęsły (<0.5). `Map Range` (From Min 0.5 → From Max 0.60–0.70) wycina z tego TYLKO wypukłe krawędzie. Potem `MixRGB` miesza kolor z jaśniejszą wersją (`kolor + 0.3`): krawędzie zużytego klocka są przetarte i jaśniejsze — dokładnie jak na prawdziwych klockach, których kanty odbijają światło po latach zabawy. **Poprawka względem starej wersji:** dawniej użyto węzła POWER, który rozjaśniał też płaskie powierzchnie; Map Range reaguje precyzyjnie tylko na kanty.

**(5) Brud w zagłębieniach (cavity).** Ten sam `Pointiness`, ale `Map Range` odwrócony (From Min ≈ 0.38–0.44 → From Max 0.5, `To Min = zuzywanie`, `To Max = 0`): wklęsłości dostają ciemniejszy kolor (`kolor × 0.6`). Fizycznie sensowne: kurz osiada w zagłębieniach, nie na wypukłościach. Intensywność maksymalna ograniczona przez `zuzywanie`.

**(6) Rysy.** Najciekawsza warstwa. Chcemy cienkich, kierunkowych smug (rys), a nie chmurki noise'u. Trik: węzeł `Texture Coordinate` (przestrzeń **Object**) → `Mapping` ze skalą anizotropową `(2–6, 25–50, 25–50)` — rozciągnięcie szumu ~10× w osi X zamienia „plamki" w „nitki". Dalej noise scale 6–10, detail 2, i **wąski ColorRamp** (progi ~0.42–0.48 i 0.52–0.58) — przepuszcza tylko wąski pas wartości szumu, wycinając pojedyncze cienkie linie. `MixRGB` nakłada rysy na wynik rampy roughness z `Fac = zuzywanie·[0.4–0.8]`.

**Podłączenie końcowe do Principled BSDF:**
- `Base Color` ← łańcuch: Mix_BaseColor → Edge_Wear → Mix_Cavity,
- `Roughness` ← **`Map_Rough`** (zacisk do 0.15–0.75). **Dlaczego zacisk:** pełny zakres 0–1 daje fragmenty lustrzane (roughness≈0) albo idealnie matowe (≈1) — oba wyglądają sztucznie; realny plastik żyje w środkowym pasie,
- `Normal` ← Bump, `Metallic` ← wartość bazowa,
- **`Coat Weight` = zuzywanie · [0.05–0.30]`** — cienka warstwa lakieru (połysk plastiku), obecna głównie przy zużyciu (odparty plastik „szkli" się nierówno).

Diagram przepływu:

```
Noise_Rough ─▶ CR_Rough ──────────────▶┐
                                       ├─▶ Mix_Rysy ─▶ Map_Rough(0.15–0.75) ─▶ Roughness
Coord ─▶ Mapping(anizo) ─▶ Noise_Rysy ─▶ CR_Rysy ─▶┘(Fac = z·0.4–0.8)
                                               
Noise_Color ─────────▶ Mix_BaseColor ─▶ Edge_Wear ─▶ Mix_Cavity ─▶ Base Color
Geometry.Pointiness ─▶ Map_Edge ──(Fac)──▶▲          ▲
Geometry.Pointiness ─▶ Map_Cavity ───────(Fac)───────┘
Noise_Bump ─▶ Bump ──────────────────────────────────────────(Normal)▶ Normal BSDF
```

Funkcja zwraca kolor bazowy — używa go później generator materiału półki (kontrast!).

## 4.6 `losuj_material_polki(kolor_klocka)`

Proceduralny materiał półki: dwa losowe kolory mieszane szumem (`Noise_Tex` scale 2–50, detail 15 — reuse węzłów po nazwie, żeby nie mnożyć ich między scenami). **Najważniejszy detal: kontrast z klockiem.** Pierwszy kolor półki losowany jest w pętli (max 10 prób) tak, by `math.dist(c1, kolor_klocka) > 0.4` — czyli półka NIE MOŻE mieć koloru zbliżonego do klocka stojącego na niej. **Dlaczego:** klocek w kolorze półki byłby trudny do wykrycia nawet dla człowieka; wymuszony kontrast dba o czytelność danych. Roughness `[0.1, 1.0]` i Metallic `[0, 0.4]` dają półkom różnorodność: od lakierowanego drewna po metalową półkę.

## 4.7 `ustaw_teksture_polki()` i `ustaw_tlo()` — tekstury z fallbackiem

**`ustaw_teksture_polki()`** — jeśli włączona opcja i katalog `data/textures/shelf/` zawiera pliki, nakłada losową teksturę obrazową na półkę: odczepia stare połączenia `Base Color`, tworzy (lub reużywa) węzły `Shelf_Tex` + `Tex_Coord`, ładuje obraz (`check_existing=True` — Blender nie duplikuje pamięci dla tej samej tekstury) i podpina UV → Color → Base Color. Zwraca `True` przy sukcesie. **Dlaczego fallback:** w głównej pętli wywołanie jest w warunku `if not ustaw_teksture_polki(): losuj_material_polki(kolor_ref)` — gdy tekstur brak albo się nie załadują, półka dostaje materiał proceduralny z 4.6. System nigdy nie zostaje bez materiału.

**`ustaw_tlo()`** — analogicznie dla świata (world): węzeł `Background` dostaje `Environment Texture` (HDRI) z siłą `[0.5, 2.0]`; przy braku tekstur — losowy kolor z siłą `[0.3, 1.5]`. HDRI dodatkowo **oświetla scenę** naturalnym światłem otoczenia; kolorowe tło to za to czysta randomizacja wizualna. (Katalog `backgrounds/` jest obecnie pusty — działa gałąź fallback.)

## 4.8 `ustaw_kamere()` — pozycja sferyczna

```python
r     = uniform(8, 22)      # odległość od półki
theta = uniform(0, 2π)      # azymut (obrót wokół pionu)
phi   = uniform(0.05, 0.85) # wysokość (kąt zenitalny)
```

Współrzędne sferyczne zamieniane na XYZ (`r·sinφ·cosθ, r·sinφ·sinθ, r·cosφ`). Następnie kierunek „od kamery do środka półki" zamieniany jest na rotację przez `to_track_quat('-Z', 'Y')`: w Blenderze kamera patrzy wzdłuż lokalnej osi **−Z**, a jej „góra" to +Y — kwaternion ustawia więc kamerę tak, by celowała w półkę bez przechylenia horyzontu. Na koniec `lens = uniform(20, 70)` mm — losowa ogniskowa symuluje różne obiektywy (szeroki kąt ↔ tele). **Dlaczego sferycznie a nie losowy punkt w prostopadłościanie:** gwarantuje równomierną pokrycie kierunków widoku przy stałym dystansie — żaden kąt nie jest faworyzowany, żadna klatka nie powstaje z wnętrza półki.

## 4.9 `losuj_swiatla()`

Najpierw usunięcie wszystkich istniejących lamp (scena zawsze zaczyna od zera — bez „duchów" poprzedniej klatki). Potem `n ∈ [1,3]` lamp:

- **Lamp 0 = zawsze duże miękkie AREA**: pozycja ±4 j.u., wysokość 5–8, `size 3–8`, energia 800–2500. To **światło kluczowe** (key light). Duża powierzchnia lampy = miękkie cienie z łagodnymi przejściami — tak wygląda oświetlenie studyjne/domowe. Wcześniej pierwszym światłem mógł być ostry POINT i cienie były twarde, CGI-owe.
- **Pozostałe** losują typ z {POINT, SUN, AREA}: SUN to „słońce" (kierunkowe, pozycja bez znaczenia — liczy się rotacja, energia 2–8), POINT to żarówka (energia 200–1500), AREA mniejsza (size 1–5).
- Barwa każdego światła lekko losowana `(0.8–1.0, 0.8–1.0, 0.7–1.0)` RGB — od neutralnej po ciepłą (kanał B celowo ma niższe minimum).

Efekt: każda klatka ma inne, ale fizycznie wiarygodne oświetlenie.

## 4.10 Kolizje AABB: `_aabb_xy`, `nachodza`

**AABB** (Axis-Aligned Bounding Box) to najmniejszy prostokąt równoległy do osi otaczający obiekt. `_aabb_xy(o)` bierze 8 rogów `bound_box` obiektu, przemnaża przez `matrix_world @ Vector(c)` (macierz świata = uwzględnia pozycję/rotację/skalę) i zwraca `(x_min, x_max, y_min, y_max)`.

`nachodza(a, b, margines=0.05)` sprawdza separację osi: prostokąty NIE nachodzą na siebie, tylko jeśli jeden kończy się przed początkiem drugiego w X lub Y (+ margines bezpieczeństwa 0.05 j.u.). Negacja tego warunku = kolizja. **Dlaczego AABB a nie odległość środkków:** klocek 2×4 jest prostokątny — okrąg okalający byłby znacznie większy niż klocek (fałszywe odrzucenia pozycji) albo mniejszy w rogach (przenikanie). AABB jest dokładny i tani (kilka porównań liczb).

## 4.11 `ORIENTACJE` i `osadz_na_polce()`

Klocek 2×4 ma dokładnie **6 stabilnych położeń**: płasko (dwie strony), do góry nogami, oraz leżenie na długim i krótkim boku (po dwie strony). Lista `ORIENTACJE` trzyma je jako pary `(rx, ry)` — obroty wokół osi X i Y: `(0,0)`, `(π,0)`, `(±π/2, 0)`, `(0, ±π/2)`. Trzecia składowa rotacji (Z) losowana dowolnie w `postaw_klocki` — obrót wokół pionu nie destabilizuje klocka. **Dlaczego lista a nie pełny losowy Euler:** losowe obroty dawałyby klocki „balansujące" w niemożliwych fizycznie pozycjach — model uczyłby się geometrii, która w realu nie występuje.

`osadz_na_polce(obj)`: odświeża drzewo sceny (`view_layer.update()` — inaczej macierze byłyby nieaktualne), znajduje najniższy róg `bound_box` w świecie i przesuwa obiekt w dół o różnicę `POLKA_TOP_Z - min_z`. Efekt: obiekt **dokładnie dotyka** wierzchu półki, niezależnie od orientacji. To rozwiązuje dawny problem „z=-1 na sztywno", przy którym klocki leżące na boku wchodziły w półkę lub unosiły się nad nią.

## 4.12 `postaw_klocki(n)` — serce generatora

Krok po kroku dla każdego z n klocków:
1. **Klonowanie**: `obj_wzor.copy()` tworzy nowy obiekt, ale `klon.data = obj_wzor.data.copy()` dodatkowo kopiuje mesh. **Dlaczego oba:** bez `data.copy()` wszystkie klony dzieliłyby tę samą siatkę — modyfikator Bevel i ewentualne zmiany wpływałyby na wszystkich naraz. Link do kolekcji (`objects.link`) sprawia, że klocek pojawia się w scenie; zdjęcie flag `hide_render/hide_viewport` go ujawnia.
2. **Rozmieszczenie**: wybór orientacji z `ORIENTACJE`, potem do **200 prób**: losowy obrót Z + pozycja `(X_MIN..X_MAX, Y_MIN..Y_MAX)`, `osadz_na_polce`, sprawdzenie kolizji ze wszystkimi już postawionymi klockami. Sukces = `break`.
3. **Pułapka językowa `for–else`**: blok `else` pętli `for` wykonuje się, gdy pętla **skończyła się bez `break`** — czyli 200 prób nie dało wolnego miejsca. Wtedy klon jest usuwany (`bpy.data.objects.remove`) razem z meshem, jeśli nikt inny go nie używa (`mesh.users == 0` — higiena pamięci), a pętla główna robi `continue`. Dzięki temu scena nigdy nie zawiera przenikających się klocków, nawet gdy „brakuje miejsca".
4. `losuj_material_klocka(klon, idx)` i dopisanie do listy zwrotnej.

Statystyka z testów (SEED=42): skip 0/20 scen, średnio ~3.45 klocka na obraz (wcześniejsza wersja: ~1.0 i 50% odrzutów — winowajcą był zbyt duży obszar losowania, naprawiony limitem `ZAKRES_XY`).

`usun_klocki(klocki)` — sprzątanie po renderze (ten sam wzorzec usuwania z kontrolą `users`).

## 4.13 `postaw_rozpraszacze(n, klocki)` — distractors

Rozpraszacze to obiekty-inne-obiekty-na-scenie: kula, sześcian, torus lub stożek (`bpy.ops.mesh.primitive_*_add` przez słownik nazw), przeskalowane do 0.1–0.4 j.u. Rozstawiane tym samym mechanizmem co klocki (200 prób), ale muszą omijać **zarówno klocki, jak i siebie nawzajem**. Każdy dostaje własny materiał: losowy kolor, roughness `[0.2, 0.9]`.

**Dlaczego są kluczowe:** bez nich model nauczyłby się reguły „wszystko kolorowe na półce = klocek". Rozpraszacze wymuszają naukę **cech kształtu** (stożek na głowie klocka, wieńce, proporcje) zamiast kontekstu. To klasyczna technika redukcji fałszywych alarmów w danych syntetycznych.

## 4.14 `ustaw_dof(klocki)` — fizyczny DOF

```python
if not klocki or random.random() < P_BRAK_DOF:
    use_dof = False          # 25% klatek całkowicie ostrych
    return
use_dof = True
srednica_otworu = uniform(1.5, 9.0)                    # mm
aperture_fstop  = max(lens / srednica_otworu, 1.2)     # f-stop = ogniskowa / otwór
focus_object    = random.choice(klocki)
```

Ostrość ustawiana jest na **losowym klocku** — czasem pierwszy plan, czasem tył, jak w realnych zdjęciach robionych bez namysłu. Zakres otworu 1.5–9 mm daje przekrój od płytkiej ostrości (portretowej) do niemal panoramicznej. Stała `P_BRAK_DOF=0.25` zapewnia, że model zobaczy też idealnie ostre ujęcia. **Historia wersji (warto znać):** wcześniej kod losował f-stop 1.4–8 bez związku z ogniskową — to nie miało sensu fizycznego; teraz rozmycie skaluje się realistycznie z doborem obiektywu z 4.8.

`ustaw_compositor()` — celowo `pass`: API compositora jest niestabilne między wersjami Blendersa (projekt na 5.x), a jego efekty (dystorsja obiektywu itd.) zastępują: losowa ogniskowa kamery + augmentacja pikselowa. Uczciwa decyzja inżynierska, łatwa do obronienia.

## 4.15 `bbox_yolo(obj)` — jak powstają etykiety GT

To funkcja, która „za darmo" daje idealne etykiety. Krok po kroku:
1. `view_layer.update()` — aktualizacja macierzy.
2. 8 rogów `bound_box` → przestrzeń świata.
3. **`bpy_extras.object_utils.world_to_camera_view`** — rzutuje każdy róg 3D na płaszczyznę kadru kamery, dając współrzędne `(x, y)` w zakresie [0,1] (plus głębokość z).
4. **Flip Y:** ekranowa oś Y rośnie w dół, więc `ys = 1 − p.y`.
5. Liczone są `orig_w/orig_h` — rozmiary boxa PRZED przycinaniem.
6. Box przycinany do kadru: `x0 = max(0, min(xs))`, `x1 = min(1, max(xs))` itd. Jeśli po przycięciu nic nie zostało (`x0 >= x1`) — obiekt w całości poza kadrem → `None`.
7. **Filtr widoczności:** jeśli widoczne jest mniej niż **20% któregoś wymiaru** boxa (`w < 0.2*orig_w` lub `h < ...`) → `None`. Odrzuca klocki wiszące na samej krawędzi kadru.
8. Zwrot `(x_center, y_center, width, height)` — dokładnie format YOLO z 3.2.

Dodatkowy filtr w głównej pętli odrzuca etykiety o szerokości/wysokości < 0.05 kadru (mikroskopijne boxy psują trening).

**Znane ograniczenie (możesz zostać o to zapytany):** bbox liczony z `bound_box` obejmuje cały klocek — także wtedy, gdy jest **w 100% zasłonięty** przez inny obiekt. Filtr zasłonięć (ray-casting od kamery) to zaplanowana, niewdrożona poprawka.

## 4.16 Geometria półki i główna pętla generacji

Przed pętlą (jednorazowo): półka dostaje materiał (jeśli nie ma), pozycjonowana jest na `(0, 0, -2)`, a z jej `bound_box` wyliczają się stałe: **`POLKA_TOP_Z`** (najwyższy róg — poziom podłogi dla klocków) oraz granice `X_MIN/X_MAX/Y_MIN/Y_MAX` = `min/max` rogów ± `KRAWEDZ_POLKI`, **dodatkowo zacięte do ±`ZAKRES_XY`**. Formuła `max(min+krawędź, −zakres)` gwarantuje poprawny przedział nawet przy bardzo dużej półce.

Główna pętla (`for i in range(NUM_SAMPLES)`):

1. **Split**: `r = random.random()`; `< 0.70` → train, `< 0.85` → val, inaczej test. Losowanie per-klatka (statystycznie ≈70/15/15, nie gwarantowane co do sztuki).
2. **Scena pusta czy nie**: `pusta = random.random() < P_PUSTA_SCENA`; jeśli nie pusta, `n_zadane = randint(MIN_KLOCKI, MAX_KLOCKI)`.
3. `postaw_klocki(n_zadane)`. Jeśli wyszło 0 klocków, choć scena **miała** mieć obiekty (wszystkie klony odrzucone przez kolizje) → `skip += 1; continue` — taka klatka nie spełnia założenia i nie trafia do datasetu. Sceny puste *celowo* (`pusta=True`) przechodzą dalej jako ramki tła.
4. `postaw_rozpraszacze(randint(0, MAX_DYSTR), klocki)`.
5. Materiał półki: próba tekstury, fallback proceduralny z kontrastem do koloru pierwszego klocka (guard: gdy brak klocków, `(0.5,)*3`).
6. Kamera → DOF → tło → światła (kolejność ważna: DOF korzysta z ogniskowej ustawionej przez kamerę).
7. **Etykiety**: `bbox_yolo` dla każdego klocka + filtr `w,h > 0.05`.
8. Render: `bpy.ops.render.render(write_still=True)` do `img_{frame:04d}.png`. Numer `frame` rośnie globalnie (nie per split) — nazwy unikalne.
9. **Plik `.txt` pisany ZAWSZE**, nawet pusty (ramka tła — patrz 3.2). Dawniej pisany tylko gdy były etykiety, przez co puste sceny były bezużyteczne dla YOLO.
10. Sprzątanie klonów i rozpraszaczy; licznik postępu co 100 klatek.

Na końcu zapisywany jest **`config.json`** (`generated`, `skipped`, liczniki splitów, rozdzielczość, klasa, silnik) — metadane datasetu przydatne w pracy do opisu eksperymentu.

---

# 5. Augmentacja

Plik: `src/augmentuj.py`. Biblioteki: **albumentations ≥ 2.0** (pipeline transformacji) + OpenCV (operacje pikselowe).

```bash
python src/augmentuj.py data/lego_dataset --kopie 1 --seed 42
```

## 5.1 Filozofia: trzy żelazne zasady

1. **Tylko piksele, zero geometrii** — nie ma obrotów/przesunięć/skalowania obrazu, więc bounding boxy z etykiet pozostają w 100% poprawne. Każda transformacja geometryczna wymagałaby przekształcania etykiet (i ryzyka błędów).
2. **Kopie `_augN`, oryginały nietknięte** — dataset rośnie ×(1+liczba_kopii); rerun jest bezpieczny, bo pliki z `_aug` w nazwie są pomijane przy kolejnym uruchomieniu.
3. **Tylko split `train`** — walidacja/test muszą mierzyć wydajność na czystych danych; augmentacja tam zafałszowałaby metryki.

Labelki kopii powstają przez `shutil.copyfile` — identyczne bajt w bajt.

## 5.2 Kolejność efektów = fizyka akwizycji obrazu

Pipeline naśladuje kolejność, w jakiej zniekształcenia powstają w PRAWDZIWYM aparacie:

```
scena → [OBIEKTYW: rozmycie, aberracje, winieta] → [SENSOR: szum] →
→ [PROCESOR ISP: jasność/kontrast, HSV, balans bieli] → [ZAPIS: kompresja JPEG]
```

Szum sensora musi być PO rozmyciu — w rzeczywistości szum powstaje już po optyce, na matrycy; odwrotna kolejność „wyczyściłaby" szum rozmyciem i dała nienaturalny efekt. Ta poprawka kolejności była jednym z wyników audytu modułu.

## 5.3 Stałe parametrów (tablica do nauki)

| Grupa | Parametry | Co ustalają |
|---|---|---|
| Szum | `_SZUM_GAUSS_SCALE=(0.02,0.08)` | Odchylenie std szumu Gaussa względem maksimum (albumentations ≥2.0: parametr `std_range`) |
| Rozmycie | `_ROZMYCIE_RUCH_I_SREDNICA=(3,7)`, `_SIGMA=(0.3,1.2)` | Rozmiar jądra blur ruchu/gaussa i sigma |
| JPEG | `_JPEG_JAKOSC=(60,95)` | Jakość kompresji symulującej zapis aparatu |
| Jasność/kontrast | `(-0.10,+0.10)`, `(-0.15,+0.15)` | Subtelne — mocne zmiany „rozjeżdżałyby" kolory GT |
| HSV | hue ±15°, sat/val ±20% | Wariacje barwy/nasycenia/jasności |
| Winieta | intensywność 0.15–0.40 | Poprawione z saturujących 0.30–0.70 (patrz 5.4) |
| Aberracja | przesunięcie ≤3 px, rozmycie 0.6 | Siła rozchodzenia kanałów R/B |
| Ekspozycja lokalna | okno 80–250 px, σ=0.25·okna, wariacja ±12% | Rozmiar/moc „plamy świetlnej" |

## 5.4 Własne efekty (funkcje pomocnicze)

**`_kompresja_jpeg(obraz, jakosc)`** — `cv2.imencode(".jpg")` do bufora w pamięci + natychmiastowe `cv2.imdecode`. Symuluje artefakty kompresji (bloki 8×8, dzwonienie przy krawędziach) bez ruszania dysku.

**`_winietowanie(obraz, intensywnosc)`** — maska ciemnienia: dla każdego piksela normalizowana odległość od środka `d_norm = dystans/√2` (dzięki temu **narożniki kadru mają dokładnie 1.0**), maska `1 − i·d_norm²`. Ciemnienie rośnie z **kwadratem** odległości — aproksymacja fizycznego prawa cos⁴ (vignetting w optyce spada z czwartą potęgą kąta). Stara wersja ciemniła liniowo od promienia 0.4–0.85, co przyciemniała aż 60–70% kadru i potrafiła zgasić rogi do czerni; nowa działa subtelnie, pełną siłę osiągając dopiero w rogach.

**`_aberracja_chromatyczna(obraz, przesuniecie, rozmycie)`** — w prawdziwym obiektywie różne barwy załamują się pod innymi kątami, więc kanały R i B minimalnie się rozjeżdżają, **coraz mocniej dalej od środka**. Kod: siatka współrzędnych `[-1,1]`, przesunięcie `dx = x·(x²+y²)·shift/szer` (radialne, ∝ r², zero w centrum), `cv2.remap` przesuwa kanał B (indeks 0!) w jedną stronę, R (indeks 2) w przeciwną, każdy uprzednio lekko rozmyty Gaussowsko (symulacja rozmycia spektralnego). **Uwaga BGR:** OpenCV czyta obrazy w kolejności B-G-R, stąd komentarze w kodzie pilnujące indeksów kanałów. Poprawka vs starą wersję: dawniej `dx ∝ x+y` (wzór diagonalny, nieradialny) i tylko jeden kanał był przesuwany.

**`_balans_bieli(obraz, temperatura)`** — temperatura barwowa 5500–7500 K względem referencyjnych 6500 K: cieplej (>6500) mnoży R przez `1+Δ·0.05` i dzieli B przez `1+Δ·0.03`; chłodniej — odwrotnie (asymetria 0.05/0.03 imituje czułość krzywych spektralnych matrycy). Δ = `(temp−6500)/100`.

**`_ekspozycja_lokalna(obraz)`** — „plama" światła/cienia: pojedynczy punkt w kadrze → maska gaussowska (GaussianBlur σ = 0.25 rozmiaru okna) → normalizacja → mnożenie jasności przez `1 ± 12%` w obszarze plamy. Symuluje np. odbicie światła od blatu albo winietujący cień palca nad obiektywem.

## 5.5 `augmentuj_obraz(sciezka_in, sciezka_out, seed=None)`

Buduje jeden pipeline `A.Compose([...], seed=seed)` i wykonuje go na obrazie wczytanym `cv2.imread`. Skład (kolejność z 5.2, prawdopodobieństwa przy każdej pozycji):

| # | Transformacja | p | Uwagi |
|---|---|---|---|
| 1 | `OneOf(MotionBlur, GaussianBlur)` | 0.35 | rozmycie ruchu kamery lub nieostrości |
| 2 | `OneOf(GaussNoise(std_range), ISONoise)` | 0.45 | szum elektroniczny / wysokich ISO (ISO ma też przesunięcie barwy) |
| 3 | `RandomBrightnessContrast` | 0.50 | `brightness_by_max=True` — skalowanie względem maksimum |
| 4 | `HueSaturationValue` | 0.45 | barwa/nasycenie/jasność |
| 5 | `Lambda(_balans_bieli)` | 0.40 | temperatura losowana raz na obraz |
| 6 | `Lambda(_winietowanie)` | 0.45 | intensywność raz na obraz |
| 7 | `Lambda(_aberracja_chromatyczna)` | 0.25 | shift 1–3 px raz na obraz |
| 8 | `Lambda(_ekspozycja_lokalna)` | 0.30 | — |
| 9 | `Lambda(_kompresja_jpeg)` | 0.40 | zawsze ostatnia (jak zapis w aparacie) |

Detale implementacyjne, które warto umieć wyjaśnić:
- **Parametry własnych efektów losowane są raz na obraz** (globalny `random`, ewentualnie zasiedowany na początku funkcji), a lambdy je „zapamiętują". Gdyby losować wewnątrz lambdy, trudniej byłoby zapewnić determinizm.
- **Lambdy mają sygnaturę `(im, **kw)`** — albumentations przekazuje do funkcji dodatkowy argument `shape`; bez `**kw` pipeline wywala się z `TypeError`.
- `A.Lambda` to oficjalny sposób wpięcia własnej funkcji pikselowej do Compose.

## 5.6 Determinizm — najtrudniejszy temat tego modułu

**Problem:** ustawienie `random.seed(42)` NIE wystarcza. Albumentations 2.x tworzy dla **każdej transformacji własne generatory losowe** (`np.random.default_rng(seed)` + `random.Random(seed)`); bez podania seeda są one inicjalizowane entropią systemową — dwa uruchomienia dają różne obrazy.

**Rozwiązanie dwupoziomowe:**
1. `A.Compose(..., seed=seed)` — seedyje wszystkie wbudowane transformacje pipeline'u.
2. **Seed per obraz**: `seed_img = (crc32(nazwa_pliku) ^ seed) % 2³¹` przekazywany do `augmentuj_obraz`. **Dlaczego tak:** gdyby wszystkie obrazy dostały ten sam seed, wszystkie miałyby IDENTYCZNĄ kombinację efektów (bezużyteczne); gdyby seedował tylko licznik, wynik zależałby od kolejności plików. CRC32 nazwy jest stabilny między uruchomieniami i platformami, a XOR z globalnym seedem pozwala wygenerować różne zestawy kopii (`--seed 43` → inne augmentacje tych samych plików).

Test akceptacyjny modułu: 2× run z `--seed 42` → PNG **bajt w bajt identyczne**; jednocześnie kopie różnych obrazów różnią się między sobą (różnorodność zachowana).

## 5.7 `augmentuj_dataset` + CLI

Iteruje po `images/train/*.png` (pomijając `*_aug*`), dla każdego: sprawdza istnienie labelki (brak → warning i pominięcie), tworzy `--kopie` par `{nazwa}_aug{k}.png/.txt`. Postęp co 100 obrazów. CLI (argparse): pozycyjny `dataset`, `--kopie N` (default 1), `--seed S` (default None = losowo). `main()` pod `if __name__ == "__main__"` — import modułu niczego nie wykonuje (w przeciwieństwie do `predict.py`, patrz 8.1).

# 6. Trening

Plik: `src/detection/train.py`. Biblioteka: **Ultralytics** (YOLO11).

```python
model = YOLO(PROJECT_ROOT / "models" / "yolo11n.pt")
model.train(data=..., epochs=40, imgsz=416, batch=32,
            cache=True, amp=True, device="mps", name="lego_nowy", exist_ok=True)
```

Koncepcja: **transfer learning**. Nie trenujemy sieci od zera — startujemy od wag `yolo11n.pt` wytrenowanych na zbiorze COCO (miliony zdjęć, 80 klas). Sieć już „umie widzieć" krawędzie, tekstury, obiekty; my tylko dostosowujemy ją do jednej klasy na naszym datasecie. Przy małym syntetycznym datasetie to różnica między modelem, który się uczy, a modelem, który nie zbiega.

Hiperparametry — co znaczy każdy:

| Parametr | Wartość | Znaczenie |
|---|---|---|
| `epochs` | 40 | Liczba pełnych przejść przez dataset. Kompromis czas/jakość |
| `imgsz` | 416 | Rozmiar, do którego YOLO skaluje każdy obraz wejściowy. Mniejszy = szybciej; render 640→416 lekko wygładza drobny szum |
| `batch` | 32 | Ile obrazów naraz w jednej iteracji treningu |
| `cache=True` | — | Obrazy trzymane w RAM — druga epoka nie czyta dysku |
| `amp=True` | — | **Automatic Mixed Precision**: część obliczeń w float16 → szybszy trening i mniej pamięci, przy pomijalnej utracie dokładności |
| `device="mps"` | — | Backend **Metal Performance Shaders** — akceleracja GPU na Apple Silicon (odpowiednik CUDA na NVIDII) |
| `name="lego_nowy"` | — | Folder wynikowy `runs/detect/lego_nowy/` |
| `exist_ok=True` | — | Nadpisuje poprzedni run o tej nazwie |

Wynikiem jest `runs/detect/.../weights/best.pt` — wagi modelu z najlepszą walidacją; to one trafiają potem do ewaluacji i kamery.

---

# 7. Ewaluacja

Pliki: `evaluate_model.py` i `model_ev.py` — **identyczne kodowo** poza ścieżką modelu (`train2` vs `train-2`). To świadomy duplikat do porównywania dwóch treningów (decyzja projektowa: zostawić oba). Omawiamy jeden.

Konfiguracja: próg confidence **0.7** (bardzo restrykcyjny — liczą się pewne detekcje), próg dopasowania **IoU 0.5**, progi akceptacji projektu: mAP ≥ 0.5 ORAZ zgodność liczenia ≥ 80%.

## 7.1 Funkcje pomocnicze

- **`load_gt_labels(path)`** — wczytuje plik `.txt` GT do tablicy `(N,5)` `[cls, xc, yc, w, h]`; pusty/brakujący plik → pusta tablica (nie błąd! sceny puste są legalne).
- **`yolo_to_corners(boxes)`** — konwersja formatu środek+wymiar → narożniki `[x1,y1,x2,y2]`: `x1 = xc − w/2`, itd. Wygodniejsze do liczenia przecięć.
- **`compute_iou(box, gt_boxes)`** — IoU jednej predykcji wobec WSZYSTKICH GT naraz (wektoryzacja NumPy, bez pętli): współrzędne przecięcia to `max` lewych i `min` prawych krawędzi; pole przecięcia z `max(0, ...)` (brak styku → 0); unia = pole A + pole B − przecięcie; dzielenie chronione `np.where(union > 0, ..., 0)`.

## 7.2 `compute_average_precision` — serce ewaluacji

Algorytm dla JEDNEGO obrazu:
1. Przypadki brzegowe: brak GT i brak predykcji → **AP = 1.0** (perfekcyjnie: nic nie ma, nic nie wykryto); tylko jedno z dwóch puste → AP = 0.0.
2. Predykcje sortowane **malejąco po score** — najpewniejsze oceniane pierwsze (tak działa ranking detektorów).
3. **Greedy matching**: dla każdej predykcji znajdowana GT z największym IoU; trafienie (TP) jeśli IoU ≥ progu ORAZ ta konkretna GT **nie została jeszcze użyta** (każdy obiekt może zostać trafiony tylko raz — drugi box na tym samym obiekcie = FP). To klasyczna zasada oceny detekcji.
4. Z binarnych TP/FP budowane skumulowane krzywe: `precision(i) = cum_tp(i)/(cum_tp+cum_fp)`, `recall(i) = cum_tp/n_gt`.
5. Krzywa PR dopełniana punktami brzegowymi (start `(0,1)`, koniec `(1,0)`), a AP = **pole pod krzywą całkowane metodą trapezów** (`np.trapezoid`).

⚠️ **Rozbieżność dokumentacyjna do obrony:** docstring funkcji mówi „metoda 11-punktowa" (stara metodologia PASCAL VOC), ale implementacja robi całkowanie trapezowe po wszystkich punktach (metoda all-point / styl COCO). Prawidłowa odpowiedź: „implementacja jest nowocześniejsza niż opis — all-point AP, jak w benchmarku COCO; docstring pozostaje do aktualizacji". Umiej to powiedzieć prosto.

## 7.3 `main()` i metryki końcowe

Dla każdego PNG z splitu test: wczytanie GT → predykcja (`conf=0.7`) → porównanie liczb obiektów (trafiona zgodność licznikowana) → AP. Na końcu:
- **mAP@0.5** = średnia AP po wszystkich obrazach,
- **zgodność liczenia %** = udział zdjęć, gdzie liczba predykcji == liczba GT (bezpośrednia miara celu „liczenia klocków"),
- status AKCEPTOWALNA / DO POPRAWY wg progów.

**Słabości metodyki, o które można zostać spytany (uczciwość > chwała):**
1. Para „puste GT + zero predykcji" daje AP = 1.0 — przy wielu pustych scenach mAP sztucznie rośnie (choć semantycznie to poprawne zachowanie).
2. Split test bywał mały (~7 obrazów przy starych generacjach) — pojedyncze pudełko zmienia wynik o ~14%.
3. `conf=0.7` mocno tnie recall — stąd historycznie niska zgodność liczenia (57%) mimo dobrego mAP (86%); to dwie różne rzeczy, umiej je rozdzielić.

---

# 8. Predykcja i detekcja na żywo

## 8.1 `predict.py` — pułapka importowa

```python
model   = YOLO(...best.pt)
results = model("data/zdjecia/zdjecie_klocka3.jpg", conf=0.25)
results[0].show()
for box in results[0].boxes: print(box.cls, box.conf, box.xyxy)
```

⚠️ Kod wykonuje się **na poziomie modułu** (nie ma `main()` ani `if __name__`). Skutek: sam import tego pliku odpala predykcję. Dla szybkiego skryptu-narzędzia to akceptowalne, ale to znany punkt do refaktoru (wpis w TODO). Wynik: okno z narysowanymi bboxami + konsola z listą `klasa / pewność / współrzędne`. Tu conf niskie (0.25) — celowo pokazuje wszystko, co model podejrzewa.

## 8.2 `camera_detect.py` — stream z kamery

```python
results = model.predict(source="1", show=True, conf=0.4, stream=True)
for r in results: pass
```

- `source="1"` — indeks urządzenia wideo (0 zwykle wbudowana kamera; "1" = podłączona zewnętrzna).
- `stream=True` — predict zwraca **generator** klatek przetwarzanych leniwie (płynne video); bez niego Ultralytics próbowałoby zbudować pełną listę wyników.
- Pętla `for r in results: pass` musi istnieć — to ona „przepompowuje" generator (okno z ramkami rysuje samo ultralytics przez `show=True`).
- `conf=0.4` — kompromis między fałszywymi alarmami a gubieniem klocków w ruchu.

To prototyp finalnego celu pracy: **liczenie = długość listy detekcji** powyżej filtra conf (docelowo osobny moduł `count_bricks.py`).

---

# 9. Uruchomienie całego pipeline'u

```bash
# 0) Środowisko (raz)
python3 -m venv venv && venv/bin/pip install -e .   # deps z pyproject.toml

# 1) Generacja datasetu (Blender headless; wpisz SEED w Simulation.py dla odtwarzalności)
/Applications/Blender.app/Contents/MacOS/Blender \
    -b src/simulation/Simulation.blend -P src/simulation/Simulation.py

# 2) Augmentacja train (dataset ×2, deterministycznie)
venv/bin/python src/augmentuj.py data/lego_dataset --kopie 1 --seed 42

# 3) Trening (wyniki: runs/detect/lego_nowy/)
venv/bin/python src/detection/train.py

# 4) Ewaluacja własna (test split)
venv/bin/python src/detection/evaluate_model.py

# 5) Predykcja na zdjęciu / kamera live
venv/bin/python src/detection/predict.py
venv/bin/python src/detection/camera_detect.py
```

Kolejność ma znaczenie: augmentacja PO generacji (działa na gotowym train), trening PO augmentacji (widzi ×2 danych), ewaluacja PO treningu (potrzebuje `best.pt`).

---

# 10. Pytania komisji + gotowe odpowiedzi (FAQ)

**Q1. Dlaczego dane syntetyczne, a nie prawdziwe zdjęcia?**
Etykietowanie ręczne tysięcy zdjęć jest kosztowne i podatne na błędy człowieka. Generator daje nieograniczoną liczbę obrazów z idealnymi, automatycznymi etykietami oraz pełną kontrolę nad różnorodnością scen. Ryzyko (domain gap) adresuję domain randomization i augmentacją fotograficzną.

**Q2. Czym jest domain randomization i gdzie ją stosuje Pan/Pani?**
Strategia: losujemy wszystkie cechy niezwiązane z istotą zadania, aż świat rzeczywisty stanie się „jednym z losowań". U mnie: losowe materiały proceduralne ze spektrum zużycia, kolory, światła (typ/liczba/energia), kamera sferycznie + losowa ogniskowa, fizyczny DOF w 75% klatek, tła, półki, rozpraszacze, sceny puste — plus augmentacja pikselowa symulująca aparat.

**Q3. Czemu EEVEE, a nie Cycles?**
EEVEE renderuje w czasie rzeczywistym (rasteryzacja) — setki razy szybciej przy tej jakości, która wystarcza do detekcji. Cycles dawałby fotorealizm, którego nie potrzebuję do nauczenia kształtu klocka, a kosztem niemożliwego czasu generacji.

**Q4. Jak powstają etykiety? Czy są dokładne?**
Z geometrii: 8 rogów siatki klocka rzutuję na płaszczyznę kamery (`world_to_camera_view`), biorę prostokąt otaczający, normalizuję do [0,1], przycinam do kadru i odrzucam klocki widoczne w mniej niż 20% wymiaru. Etykiety są więc matematycznie idealne względem geometrii; ograniczeniem jest to, że klocek całkowicie zasłonięty nadal dostałby etykietę — filtr zasłonięć (ray-casting) to zaplanowana poprawka.

**Q5. Co się dzieje z klockiem częściowo poza kadrem?**
Box przycinam do kadru; jeśli po przycięciu zostaje mniej niż 20% któregokolwiek wymiaru, etykieta jest odrzucana. Model nie uczy się więc na „pół-klockach" z przypadkową etykietą.

**Q6. Po co sceny bez żadnego klocka?**
10% datasetu to ramki tła: pusty plik `.txt` to dla YOLO poprawna informacja „tu nic nie ma". Takie obrazy uczą sieć redukować fałszywe alarmy — inaczej model wykrywałby klocki wszędzie.

**Q7. Dlaczego kolizje sprawdzasz AABB, a nie odległością środków?**
Klocek 2×4 jest prostokątny; okrąg okalający byłby za duży (fałszywe odrzucenia dobrych pozycji) albo przenikałby w rogach. AABB w XY to dokładny i tani test separacji osi z marginesem 5%.

**Q8. Po co rozpraszacze?**
Bez nich model uczy się heurystyki „kolorowe na półce = klocek". Losowe prymitywy (kula/sześcian/torus/stożek) zmuszają sieć do rozpoznawania właściwego kształtu klocka — klasyczna redukcja false positives.

**Q9. Czemu klocki mają tylko 6 orientacji, a nie dowolny obrót?**
Tylko te 6 położeń jest stabilnych fizycznie (środkiem ciężkości nad podstawą). Losowe Eulery tworzyłyby klocki balansujące na kantach — geometrię, której w rzeczywistości nie zobaczę.

**Q10. Czemu augmentacja tylko na train?**
Walidacja/test mierzą realną wydajność; dodanie tam zniekształceń dałoby zawyżone lub zaniżone, ale w każdym razie nieczytelne metryki. Standard branżowy: augmentować wyłącznie trening.

**Q11. Czemu kopie `_augN`, a nie modyfikacja istniejących plików?**
Oryginały pozostają nietknięte (odtwarzalność eksperymentu), rerun jest bezpieczny (pliki `_aug` są pomijane), a labelki mogę skopiować 1:1, bo transformacje są czysto pikselowe.

**Q12. Jak zapewnia Pan/Pani odtwarzalność?**
Generator: `random.seed(SEED)` — cały układ scen odtwarza się z jednej liczby. Augmentacja: seed per transformacja w albumentations 2.x (`Compose(seed=...)`) plus pochodny seed per obraz `crc32(nazwa)^seed` — dwa uruchomienia z tym samym seedem dają identyczne bajty PNG, ale kolejne obrazy mają różne efekty.

**Q13. Jak działa głębia ostrości w generatorze?**
Fizycznie: f-stop = ogniskowa [mm] / średnica otworu [mm]; losuję otwór 1.5–9 mm, więc rozmycie skaluje się realistycznie z wylosowanym obiektywem. Ostrość ustawiam na losowym klocku; 25% klatek renderuję całkowicie ostrych.

**Q14. Czym jest transfer learning i dlaczego go stosuje Pan/Pani?**
Startuję od wag YOLO11n wytrenowanych na COCO — sieć zna już ogólne cechy wizualne. Na małym syntetycznym datasetie trening od zera nie zbiegłby się sensownie; fine-tuning daje dobry model w kilkadziesiąt epok.

**Q15. Co to jest mAP@0.5 i jak Pan/Pani go liczy?**
Średnia pole pod krzywą Precision–Recall przy progu dopasowania IoU 0.5. Implementowałem to sam: sortowanie predykcji po score, zachłanne dopasowanie do GT (każdy obiekt trafiony maks. raz — dublet to FP), krzywa skumulowana, całkowanie trapezowe (metoda all-point jak w COCO).

**Q16. Miał Pan/Pani wysoki mAP, ale słabe liczenie. Dlaczego?**
To różne metryki: mAP nagradza jakość rankingów detekcji, zgodność liczenia wymaga IDEALNEJ liczby boxów. Przy conf=0.7 każde przeoczone lub zdublowane pudełko psuje całą klatkę. Dodatkowo stare generacje miały mały test set (7 obrazów) — jedno pomyłka zmieniała wynik o ~14%. Nowy generator (więcej obiektów na obraz, zero skipów) poprawia obie metryki.

**Q17. Co oznaczają AMP i MPS w treningu?**
AMP = mixed precision (część operacji w float16) — szybszy trening i mniejsze zużycie pamięci przy pomijalnej utracie jakości. MPS = backend Metal na Apple Silicon — GPU-acceleracja zamiast CPU.

**Q18. Co by Pan/Pani poprawił(a)? (pytanie niemal pewne)**
Kolejność wg priorytetu: 1) filtr widoczności zasłoniętych klocków (ray-casting od kamery), 2) HDRI-tła (katalog jest pusty, działa fallback kolorystyczny), 3) większy, zbalansowany split test i naprawa ewaluacji pustych scen, 4) osobny moduł liczenia `count_bricks.py` z trackingiem między klatkami, 5) refaktor `predict.py` pod `__main__`.

---

*Dokument opisuje stan kodu z 2026-08-23. Przy zmianach w kodzie aktualizuj odpowiednie sekcje — to Twój materiał do nauki, jego wiarygodność = Twoja pewność na obronie.*


