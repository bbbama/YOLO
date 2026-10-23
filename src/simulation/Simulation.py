import glob
import json
import math
import os
import random

import bpy
import bpy_extras
import mathutils

# ============================================================
# KONFIGURACJA
# ============================================================
CLASS_ID = 0
NUM_SAMPLES = 10
MIN_KLOCKI = 1
MAX_KLOCKI = 10
MAX_DYSTR = 8  # rozpraszacze (distractor objects)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# reszta idzie do test

NUM_LIGHTS_MIN = 1
NUM_LIGHTS_MAX = 3

P_BRAK_DOF = 0.25     # udział zdjęć w pełni ostrych (bez DOF)
P_PUSTA_SCENA = 0.10  # udział scen bez klocków (tło dla YOLO)
# wpisz liczbę (np. 42) dla pełnej odtwarzalności układu scen
SEED = None
KRAWEDZ_POLKI = 0.3   # margines od krawędzi półki przy losowaniu pozycji
# maksymalna odległość od środka półki (klocki blisko = dobrze widoczne)
ZAKRES_XY = 2.0

USE_TEX_BG = True     # tekstura tła sceny
USE_TEX_SHELF = True  # tekstura półki

# True = usuń stare obrazy/etykiety przed generacją (brak mieszania
# przebiegów); False = dogeneruj do istniejącego datasetu (numeracja
# od ostatniej klatki)
WYCZYSC_DATASET = True
# minimalny wymiar bboxa w pikselach (mniejsze = praktycznie niewidoczne)
MIN_BBOX_PX = 2

script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))
BASE_DIR = os.path.join(PROJECT_ROOT, "data", "lego_dataset")
TEX_BG_DIR = os.path.join(PROJECT_ROOT, "data", "textures", "backgrounds")
TEX_SHELF_DIR = os.path.join(PROJECT_ROOT, "data", "textures", "shelf")

# Nadpisanie z linii poleceń (szybkie testy na małej liczbie klatek), np.:
#   LEGO_NUM_SAMPLES=3 LEGO_OUT_DIR=/tmp/test_ds \
#       blender -b Simulation.blend --python Simulation.py
NUM_SAMPLES = int(os.environ.get("LEGO_NUM_SAMPLES", NUM_SAMPLES))
BASE_DIR = os.path.abspath(os.environ.get("LEGO_OUT_DIR", BASE_DIR))
if "LEGO_SEED" in os.environ:
    SEED = int(os.environ["LEGO_SEED"])

if SEED is not None:
    random.seed(SEED)


# ============================================================
# POBRANIE OBIEKTÓW ZE SCENY
# ============================================================
obj_wzor = bpy.data.objects["LEGO-2X4-L"]  # wzór klocka - tylko do kopiowania
obj_polka = bpy.data.objects["Cube2"]      # półka
cam = bpy.data.objects["Camera"]

obj_wzor.hide_render = True
obj_wzor.hide_viewport = True

# Obłaskawienie krawędzi klocka: realistyczne refleksy na kantach
# (modyfikator dziedziczony przez wszystkie klony z postaw_klocki)
bevel = (obj_wzor.modifiers.get("Bevel")
         or obj_wzor.modifiers.new("Bevel", 'BEVEL'))
bevel.width = max(obj_wzor.dimensions) * random.uniform(0.005, 0.015)
bevel.segments = 2

scene = bpy.context.scene


# ============================================================
# USTAWIENIA RENDERERA
# ============================================================
scene.render.engine = 'BLENDER_EEVEE'
scene.render.resolution_x = 640
scene.render.resolution_y = 640
scene.render.resolution_percentage = 100
scene.render.filter_size = 1.5
scene.eevee.taa_render_samples = 128
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGB'
scene.render.image_settings.compression = 0


# ============================================================
# TWORZENIE KATALOGÓW I PLIKU YAML
# ============================================================
SPLITY = ["train", "val", "test"]

if WYCZYSC_DATASET:
    # tylko pliki wygenerowane przez skrypt (obrazy, etykiety, cache YOLO)
    for split in SPLITY:
        for wzorzec in (os.path.join(BASE_DIR, "images", split, "*.png"),
                        os.path.join(BASE_DIR, "labels", split, "*.txt")):
            for p in glob.glob(wzorzec):
                os.remove(p)
    for p in glob.glob(os.path.join(BASE_DIR, "labels", "*.cache")):
        os.remove(p)

for split in SPLITY:
    os.makedirs(os.path.join(BASE_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "labels", split), exist_ok=True)

# path bezwzględny — YOLO rozwiązuje względny path od katalogu uruchomienia
with open(os.path.join(BASE_DIR, "data.yaml"), "w") as f:
    f.write(
        f"path: {BASE_DIR}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n"
        f"\nnc: 1\nnames: ['lego_2x4']\n"
    )


def pierwsza_wolna_klatka():
    """Numer następnej klatki — przy dogenerowaniu nie nadpisujemy starych."""
    wzorzec = "img_[0-9][0-9][0-9][0-9].png"
    numery = [int(os.path.basename(p)[4:8])
              for s in SPLITY
              for p in glob.glob(os.path.join(BASE_DIR, "images", s, wzorzec))]
    return max(numery) + 1 if numery else 0


# ============================================================
# ŁADOWANIE TEKSTUR
# ============================================================
def wczytaj_tekstury(katalog):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tga", "*.hdr", "*.exr")
    pliki = []
    for e in exts:
        pliki += glob.glob(os.path.join(katalog, e))
        pliki += glob.glob(os.path.join(katalog, e.upper()))
    return pliki


bg_textures = wczytaj_tekstury(TEX_BG_DIR)
shelf_textures = wczytaj_tekstury(TEX_SHELF_DIR)
print(f"Tekstury tła: {len(bg_textures)}, półki: {len(shelf_textures)}")


# ============================================================
# POMOCNIK: PRINCIPLED BSDF
# ============================================================
def nowy_material(obj, nazwa):
    """Przypisuje świeży materiał do obiektu i zwraca węzeł BSDF."""
    mat = bpy.data.materials.new(nazwa)
    obj.active_material = mat
    return mat.node_tree.nodes.get("Principled BSDF")


# ============================================================
# MATERIAŁY
# ============================================================
def losuj_material_klocka(obj, idx):
    """Proceduralny materiał klocka ze spektrum zużycia.

    Warstwy: plamy koloru, rysy, brud w zagłębieniach, połysk.
    """
    node = nowy_material(obj, f"Mat_Klocek_{idx}")
    if not node:
        return (0.5, 0.5, 0.5)

    mat = obj.active_material
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # -- losowe wartości bazowe --
    kolor = (random.random(), random.random(), random.random(), 1.0)
    bazowy_metallic = random.uniform(0.0, 0.1)
    # 0 = klocek jak nowy (gładki), 1 = mocno używany (teksturowany)
    zuzywanie = random.random()

    # ================================================================
    # 1. Noise → ColorRamp → Roughness  (mikro-zarysowania)
    # ================================================================
    n_rough = nodes.new("ShaderNodeTexNoise")
    n_rough.location = (-750, 150)
    n_rough.name = "Noise_Rough"
    n_rough.inputs["Scale"].default_value = random.uniform(60.0, 120.0)
    n_rough.inputs["Detail"].default_value = 8.0
    n_rough.inputs["Roughness"].default_value = 0.6

    cr_rough = nodes.new("ShaderNodeValToRGB")
    cr_rough.location = (-500, 150)
    cr_rough.name = "CR_Rough"
    # rampa: ciemne → niskie roughness, jasne → wysokie roughness
    cr_rough.color_ramp.elements[0].position = 0.35 - zuzywanie * 0.1
    cr_rough.color_ramp.elements[1].position = 0.65 + zuzywanie * 0.1

    links.new(n_rough.outputs["Fac"], cr_rough.inputs["Fac"])

    # ================================================================
    # 2. Noise → Bump  (delikatne nierówności normalnych)
    # ================================================================
    n_bump_tex = nodes.new("ShaderNodeTexNoise")
    n_bump_tex.location = (-750, -300)
    n_bump_tex.name = "Noise_Bump"
    n_bump_tex.inputs["Scale"].default_value = random.uniform(80.0, 150.0)
    n_bump_tex.inputs["Detail"].default_value = 6.0

    n_bump = nodes.new("ShaderNodeBump")
    n_bump.location = (-500, -300)
    n_bump.name = "Bump"
    n_bump.inputs["Strength"].default_value = (
        0.02 + zuzywanie * random.uniform(0.08, 0.23))
    n_bump.inputs["Distance"].default_value = 0.01
    links.new(n_bump_tex.outputs["Fac"], n_bump.inputs["Height"])

    # ================================================================
    # 3. Drugi Noise (duża skala) → Mix Color z Base Color
    # ================================================================
    n_color = nodes.new("ShaderNodeTexNoise")
    n_color.location = (-750, 0)
    n_color.name = "Noise_Color"
    n_color.inputs["Scale"].default_value = random.uniform(2.0, 8.0)
    n_color.inputs["Detail"].default_value = 4.0

    mix_color = nodes.new("ShaderNodeMixRGB")
    mix_color.location = (-450, 0)
    mix_color.name = "Mix_BaseColor"
    mix_color.blend_type = 'MIX'
    mix_color.inputs["Color1"].default_value = kolor
    links.new(n_color.outputs["Color"], mix_color.inputs["Color2"])
    mix_color.inputs["Fac"].default_value = min(
        random.uniform(0.05, 0.15)
        + zuzywanie * random.uniform(0.10, 0.25),
        0.45)

    # ================================================================
    # 4. Geometry → Pointiness  (przetarcia na krawędziach)
    # ================================================================
    n_geom = nodes.new("ShaderNodeNewGeometry")
    n_geom.location = (-750, -500)
    n_geom.name = "Geometry"

    # Map Range: rozjaśniamy TYLKO wypukłe krawędzie (pointiness > 0.5)
    map_edge = nodes.new("ShaderNodeMapRange")
    map_edge.location = (-550, -500)
    map_edge.name = "Map_Edge"
    map_edge.inputs["From Min"].default_value = 0.5
    map_edge.inputs["From Max"].default_value = random.uniform(0.60, 0.70)
    links.new(n_geom.outputs["Pointiness"], map_edge.inputs["Value"])

    # rozjaśnienie krawędzi: mix z jaśniejszym odcieniem
    mix_edge = nodes.new("ShaderNodeMixRGB")
    mix_edge.location = (-300, -450)
    mix_edge.name = "Edge_Wear"
    mix_edge.blend_type = 'MIX'
    links.new(mix_color.outputs["Color"], mix_edge.inputs["Color1"])
    # jaśniejsza wersja koloru (rozjaśnienie o ~30 %)
    jasny = tuple(min(c + 0.3, 1.0) for c in kolor[:3]) + (1.0,)
    mix_edge.inputs["Color2"].default_value = jasny
    links.new(map_edge.outputs["Result"], mix_edge.inputs["Fac"])

    # ================================================================
    # 5. Pointiness odwrócony → brud w zagłębieniach (cavity)
    # ================================================================
    map_cav = nodes.new("ShaderNodeMapRange")
    map_cav.location = (-550, -650)
    map_cav.name = "Map_Cavity"
    # wklęsłości (pointiness < 0.5) dostają brud;
    # maks. intensywność = zuzywanie
    map_cav.inputs["From Min"].default_value = random.uniform(0.38, 0.44)
    map_cav.inputs["From Max"].default_value = 0.5
    map_cav.inputs["To Min"].default_value = zuzywanie
    map_cav.inputs["To Max"].default_value = 0.0
    links.new(n_geom.outputs["Pointiness"], map_cav.inputs["Value"])

    mix_cav = nodes.new("ShaderNodeMixRGB")
    mix_cav.location = (-150, -600)
    mix_cav.name = "Mix_Cavity"
    mix_cav.blend_type = 'MIX'
    links.new(mix_edge.outputs["Color"], mix_cav.inputs["Color1"])
    ciemny = tuple(c * 0.6 for c in kolor[:3]) + (1.0,)
    mix_cav.inputs["Color2"].default_value = ciemny
    links.new(map_cav.outputs["Result"], mix_cav.inputs["Fac"])

    # ================================================================
    # 6. Rysy: noise rozciągnięty mappingiem → cienkie smugi w roughness
    # ================================================================
    coord = nodes.new("ShaderNodeTexCoord")
    coord.location = (-1050, -800)
    coord.name = "Coord_Rysy"

    mapa = nodes.new("ShaderNodeMapping")
    mapa.location = (-900, -800)
    mapa.name = "Map_Rysy"
    baza = random.uniform(25.0, 50.0)
    mapa.inputs["Scale"].default_value = (random.uniform(2.0, 6.0), baza, baza)
    links.new(coord.outputs["Object"], mapa.inputs["Vector"])

    n_rys = nodes.new("ShaderNodeTexNoise")
    n_rys.location = (-750, -800)
    n_rys.name = "Noise_Rysy"
    n_rys.inputs["Scale"].default_value = random.uniform(6.0, 10.0)
    n_rys.inputs["Detail"].default_value = 2.0
    links.new(mapa.outputs["Vector"], n_rys.inputs["Vector"])

    cr_rys = nodes.new("ShaderNodeValToRGB")
    cr_rys.location = (-550, -800)
    cr_rys.name = "CR_Rysy"
    # wąska rampa wycina z noise'u tylko cienkie linie
    cr_rys.color_ramp.elements[0].position = random.uniform(0.42, 0.48)
    cr_rys.color_ramp.elements[1].position = random.uniform(0.52, 0.58)

    links.new(n_rys.outputs["Fac"], cr_rys.inputs["Fac"])

    mix_rys = nodes.new("ShaderNodeMixRGB")
    mix_rys.location = (-300, 150)
    mix_rys.name = "Mix_Rysy"
    mix_rys.blend_type = 'MIX'
    links.new(cr_rough.outputs["Color"], mix_rys.inputs["Color1"])
    links.new(cr_rys.outputs["Color"], mix_rys.inputs["Color2"])
    # rysy widoczne proporcjonalnie do zużycia (jasne rysy = matowe ściegi)
    mix_rys.inputs["Fac"].default_value = zuzywanie * random.uniform(0.4, 0.8)

    # ================================================================
    # Podłączenie do Principled BSDF
    # ================================================================
    links.new(mix_cav.outputs["Color"], node.inputs["Base Color"])

    # Roughness: rampa + rysy, zacisk do realistycznego zakresu plastiku
    # (pełne 0–1 daje fragmenty lustrzane lub idealnie matowe)
    map_rough = nodes.new("ShaderNodeMapRange")
    map_rough.location = (-100, 150)
    map_rough.name = "Map_Rough"
    map_rough.inputs["To Min"].default_value = 0.15
    map_rough.inputs["To Max"].default_value = 0.75
    links.new(mix_rys.outputs["Color"], map_rough.inputs["Value"])
    links.new(map_rough.outputs["Result"], node.inputs["Roughness"])

    # Normal: bump z noise'a
    links.new(n_bump.outputs["Normal"], node.inputs["Normal"])

    node.inputs["Metallic"].default_value = bazowy_metallic

    # Plastikowy połysk (lakier bezbarwny) — tylko przy zużyciu
    if "Coat Weight" in node.inputs:
        node.inputs["Coat Weight"].default_value = (
            zuzywanie * random.uniform(0.05, 0.30))

    # Estetyczny układ: shader i output na końcu
    node.location = (250, 0)
    out = nodes.get("Material Output")
    if out:
        out.location = (550, 0)

    return kolor[:3]


def losuj_material_polki(kolor_klocka):
    """Proceduralny szum na półce; kontrast z kolorem klocka."""
    mat = obj_polka.active_material
    if not mat or not mat.node_tree:
        return
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    node = nodes.get("Principled BSDF")
    if not node:
        return

    # Odepnij stare połączenie Base Color
    for lnk in list(node.inputs["Base Color"].links):
        links.remove(lnk)

    # Kolor z minimalnym kontrastem 0.4 od koloru klocka
    for _ in range(10):
        c1 = (random.random(), random.random(), random.random(), 1.0)
        if math.dist(c1[:3], kolor_klocka) > 0.4:
            break
    c2 = (random.random(), random.random(), random.random(), 1.0)

    noise = nodes.get("Noise_Tex") or nodes.new("ShaderNodeTexNoise")
    noise.name = "Noise_Tex"
    noise.inputs["Scale"].default_value = random.uniform(2.0, 50.0)
    noise.inputs["Detail"].default_value = 15.0

    mix = nodes.get("Mix_Col") or nodes.new("ShaderNodeMixRGB")
    mix.name = "Mix_Col"
    mix.inputs["Color1"].default_value = c1
    mix.inputs["Color2"].default_value = c2
    mix.inputs["Fac"].default_value = random.uniform(0.1, 0.6)

    links.new(noise.outputs["Fac"], mix.inputs["Fac"])
    links.new(mix.outputs["Color"], node.inputs["Base Color"])

    node.inputs["Roughness"].default_value = random.uniform(0.1, 1.0)
    node.inputs["Metallic"].default_value = random.uniform(0.0, 0.4)


def ustaw_teksture_polki():
    """Nakłada losową teksturę na półkę; zwraca True jeśli się udało."""
    if not USE_TEX_SHELF or not shelf_textures:
        return False
    mat = obj_polka.active_material
    if not mat or not mat.node_tree:
        return False
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    node = nodes.get("Principled BSDF")
    if not node:
        return False

    for lnk in list(node.inputs["Base Color"].links):
        links.remove(lnk)

    tex = nodes.get("Shelf_Tex") or nodes.new("ShaderNodeTexImage")
    tex.name = "Shelf_Tex"
    coord = nodes.get("Tex_Coord") or nodes.new("ShaderNodeTexCoord")
    coord.name = "Tex_Coord"

    try:
        tex.image = bpy.data.images.load(random.choice(shelf_textures),
                                         check_existing=True)
        links.new(coord.outputs["UV"], tex.inputs["Vector"])
        links.new(tex.outputs["Color"], node.inputs["Base Color"])
        return True
    except Exception:
        return False


def ustaw_tlo():
    """Losuje tło: tekstura HDRI lub kolor proceduralny."""
    world = scene.world
    if not world or not world.node_tree:
        return
    nodes = world.node_tree.nodes
    bg = nodes.get("Background")
    if not bg:
        return

    if USE_TEX_BG and bg_textures:
        tex = nodes.get("Bg_Tex") or nodes.new("ShaderNodeTexEnvironment")
        tex.name = "Bg_Tex"
        try:
            tex.image = bpy.data.images.load(random.choice(bg_textures),
                                             check_existing=True)
            world.node_tree.links.new(tex.outputs["Color"],
                                      bg.inputs["Color"])
            bg.inputs["Strength"].default_value = random.uniform(0.5, 2.0)
            return
        except Exception:
            pass

    # fallback – losowy kolor
    bg.inputs["Color"].default_value = (random.random(), random.random(),
                                        random.random(), 1.0)
    bg.inputs["Strength"].default_value = random.uniform(0.3, 1.5)


# ============================================================
# KAMERA
# ============================================================
def ustaw_kamere():
    """Losowa pozycja sferyczna, skierowana na półkę + losowa ogniskowa."""
    r = random.uniform(8, 22)
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(0.05, 0.85)
    cam.location = mathutils.Vector((
        r * math.sin(phi) * math.cos(theta),
        r * math.sin(phi) * math.sin(theta),
        r * math.cos(phi),
    ))
    kierunek = obj_polka.location - cam.location
    cam.rotation_euler = kierunek.to_track_quat('-Z', 'Y').to_euler()

    # Losowa ogniskowa (zastępuje/wspiera lens distortion)
    cam.data.lens = random.uniform(20, 70)


# ============================================================
# ŚWIATŁA
# ============================================================
def losuj_swiatla():
    # Usuń stare (obiekty + osierocone dane świateł)
    for o in [o for o in bpy.data.objects if o.type == 'LIGHT']:
        bpy.data.objects.remove(o, do_unlink=True)
    for dane in [d for d in bpy.data.lights if d.users == 0]:
        bpy.data.lights.remove(dane)

    n = random.randint(NUM_LIGHTS_MIN, NUM_LIGHTS_MAX)
    for i in range(n):
        # pierwsze światło = duże miękkie AREA jako główne (miękkie cienie)
        typ = 'AREA' if i == 0 else random.choice(['POINT', 'SUN', 'AREA'])
        bpy.ops.object.light_add(type=typ)
        lt = bpy.context.active_object
        lt.name = f"Light_{i}"

        if typ == 'SUN':
            lt.location = (0, 0, 10)
            lt.rotation_euler = (random.uniform(-0.5, 0.5),
                                 random.uniform(-0.5, 0.5),
                                 random.uniform(0, 2 * math.pi))
            lt.data.energy = random.uniform(2, 8)
        elif i == 0:
            # kluczowe AREA: duże i wysoko nad półką
            lt.location = (random.uniform(-4, 4), random.uniform(-4, 4),
                           random.uniform(5, 8))
            lt.data.size = random.uniform(3, 8)
            lt.data.energy = random.uniform(800, 2500)
        else:
            lt.location = (random.uniform(-5, 5), random.uniform(-5, 5),
                           random.uniform(2, 8))
            lt.data.energy = random.uniform(200, 1500)
            if typ == 'AREA':
                lt.data.size = random.uniform(1, 5)

        lt.data.color = (random.uniform(0.8, 1.0), random.uniform(0.8, 1.0),
                         random.uniform(0.7, 1.0))


# ============================================================
# SPRAWDZANIE KOLIZJI
# ============================================================
def _aabb_xy(o):
    """AABB obiektu w płaszczyźnie XY: (x0, x1, y0, y1)."""
    pts = [o.matrix_world @ mathutils.Vector(c) for c in o.bound_box]
    return (min(p.x for p in pts), max(p.x for p in pts),
            min(p.y for p in pts), max(p.y for p in pts))


def nachodza(a, b, margines=0.05):
    """True jeśli prostokąty (AABB w XY) obiektów a i b zachodzą na siebie."""
    ax0, ax1, ay0, ay1 = _aabb_xy(a)
    bx0, bx1, by0, by1 = _aabb_xy(b)
    return not (ax1 + margines < bx0 or bx1 + margines < ax0
                or ay1 + margines < by0 or by1 + margines < ay0)


# ============================================================
# OSADZANIE NA PÓŁCE
# ============================================================
# stabilne położenia klocka: płasko / do góry nogami / na długim boku /
# na krótkim boku
ORIENTACJE = [
    (0, 0), (math.pi, 0),
    (math.pi / 2, 0), (-math.pi / 2, 0),
    (0, math.pi / 2), (0, -math.pi / 2),
]


def osadz_na_polce(obj):
    """Zsuwa obiekt w dół tak, by dotykał powierzchni półki."""
    bpy.context.view_layer.update()
    min_z = min((obj.matrix_world @ mathutils.Vector(c)).z
                for c in obj.bound_box)
    obj.location.z += POLKA_TOP_Z - min_z


# ============================================================
# KLOCKI
# ============================================================
def postaw_klocki(n):
    """Klonuje wzór n razy i rozmieszcza bez kolizji.

    Zwraca (obiekty, kolory RGB).
    """
    klocki = []
    kolory = []
    for idx in range(n):
        klon = obj_wzor.copy()
        klon.data = obj_wzor.data.copy()
        bpy.context.collection.objects.link(klon)
        klon.hide_render = klon.hide_viewport = False

        rx, ry = random.choice(ORIENTACJE)
        for _ in range(200):
            klon.rotation_euler = (rx, ry, random.uniform(0, 2 * math.pi))
            klon.location = mathutils.Vector((random.uniform(X_MIN, X_MAX),
                                              random.uniform(Y_MIN, Y_MAX),
                                              0))
            osadz_na_polce(klon)
            if not any(nachodza(klon, k) for k in klocki):
                break
        else:
            # Brak miejsca – usuń klon
            mesh = klon.data
            bpy.data.objects.remove(klon, do_unlink=True)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
            continue

        kolory.append(losuj_material_klocka(klon, idx))
        klocki.append(klon)
    return klocki, kolory


def usun_klocki(klocki):
    for k in klocki:
        mesh = k.data
        bpy.data.objects.remove(k, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def usun_osierocone_materialy():
    """Usuwa nieużywane materiały klocków i rozpraszaczy.

    Materiały tworzone są co klatkę — bez sprzątania rośnie pamięć.
    """
    prefiksy = ("Mat_Klocek_", "Mat_Dyst_")
    for m in [m for m in bpy.data.materials
              if m.users == 0 and m.name.startswith(prefiksy)]:
        bpy.data.materials.remove(m)


# ============================================================
# ROZPRASZACZE
# ============================================================
def postaw_rozpraszacze(n, klocki):
    """Tworzy n losowych kształtów omijających klocki. Zwraca listę."""
    rozpraszacze = []
    for i in range(n):
        shape = random.choice(['SPHERE', 'CUBE', 'TORUS', 'CONE'])
        getattr(bpy.ops.mesh, {
            'SPHERE': 'primitive_uv_sphere_add',
            'CUBE': 'primitive_cube_add',
            'TORUS': 'primitive_torus_add',
            'CONE': 'primitive_cone_add',
        }[shape])()

        obj = bpy.context.active_object
        obj.name = f"Dystraktor_{i}"
        obj.scale = mathutils.Vector((random.uniform(0.1, 0.4),) * 3)

        for _ in range(200):
            obj.location = mathutils.Vector((random.uniform(X_MIN, X_MAX),
                                             random.uniform(Y_MIN, Y_MAX), 0))
            osadz_na_polce(obj)
            if (not any(nachodza(obj, k) for k in klocki)
                    and not any(nachodza(obj, r) for r in rozpraszacze)):
                break
        else:
            bpy.data.objects.remove(obj, do_unlink=True)
            continue

        node = nowy_material(obj, f"Mat_Dyst_{i}")
        if node:
            node.inputs["Base Color"].default_value = (
                random.random(), random.random(), random.random(), 1.0)
            node.inputs["Roughness"].default_value = random.uniform(0.2, 0.9)

        rozpraszacze.append(obj)
    return rozpraszacze


def usun_rozpraszacze(lista):
    for obj in lista:
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


# ============================================================
# EFEKTY DODATKOWE
# ============================================================
def ustaw_dof(klocki):
    """Realistyczny DOF: fizyczny otwór obiektywu + losowy punkt ostrości."""
    if not klocki or random.random() < P_BRAK_DOF:
        cam.data.dof.use_dof = False
        return
    cam.data.dof.use_dof = True
    srednica_otworu = random.uniform(1.5, 9.0)  # mm - fizyczny otwór
    cam.data.dof.aperture_fstop = max(cam.data.lens / srednica_otworu, 1.2)
    cam.data.dof.focus_object = random.choice(klocki)


def ustaw_compositor():
    """Efekty optyczne bez Compositora - tylko ogniskowa kamery."""
    # Compositor API jest niestabilne w Blenderze 5.x - pomijamy
    # Efekt lens distortion zastępuje losowa ogniskowa w ustaw_kamere()
    pass


# ============================================================
# BOUNDING BOX YOLO
# ============================================================
def bbox_yolo(obj):
    """Zwraca (cx, cy, w, h) w [0,1] lub None jeśli obiekt poza kadrem.

    Rzutuje wierzchołki siatki (po modyfikatorach, np. Bevel), a nie
    8 narożników bound_box — prostopadłościan obrócony w perspektywie
    daje za duży bbox.
    Obiekty częściowo w kadrze dostają bbox przycięty do kadru.
    """
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.to_mesh()
    try:
        mw = obj_eval.matrix_world
        pts = [bpy_extras.object_utils.world_to_camera_view(scene, cam,
                                                            mw @ v.co)
               for v in mesh.vertices]
    finally:
        obj_eval.to_mesh_clear()

    pts = [p for p in pts if p.z > 0]  # tylko wierzchołki przed kamerą
    if not pts:
        return None
    xs = [p.x for p in pts]
    ys = [1.0 - p.y for p in pts]

    x0, x1 = max(0.0, min(xs)), min(1.0, max(xs))
    y0, y1 = max(0.0, min(ys)), min(1.0, max(ys))

    w, h = x1 - x0, y1 - y0
    w_px = w * scene.render.resolution_x
    h_px = h * scene.render.resolution_y
    if w_px < MIN_BBOX_PX or h_px < MIN_BBOX_PX:
        return None

    return (x0 + w / 2, y0 + h / 2, w, h)


# ============================================================
# GŁÓWNA PĘTLA
# ============================================================
# Upewnij się, że półka ma własny materiał
if obj_polka.active_material is None:
    nowy_material(obj_polka, "Mat_Polka")
obj_polka.location = (0, 0, -2)

# Geometria półki: wierzch + granice losowania (raz, przed pętlą)
bpy.context.view_layer.update()
_rogi_polki = [obj_polka.matrix_world @ mathutils.Vector(c)
               for c in obj_polka.bound_box]
POLKA_TOP_Z = max(p.z for p in _rogi_polki)
_xs = [p.x for p in _rogi_polki]
_ys = [p.y for p in _rogi_polki]
X_MIN = max(min(_xs) + KRAWEDZ_POLKI, -ZAKRES_XY)
X_MAX = min(max(_xs) - KRAWEDZ_POLKI, ZAKRES_XY)
Y_MIN = max(min(_ys) + KRAWEDZ_POLKI, -ZAKRES_XY)
Y_MAX = min(max(_ys) - KRAWEDZ_POLKI, ZAKRES_XY)

counts = {"train": 0, "val": 0, "test": 0}
frame0 = pierwsza_wolna_klatka()
frame = frame0
skip = 0

for i in range(NUM_SAMPLES):
    # Wylosuj split
    r = random.random()
    if r < TRAIN_RATIO:
        split = "train"
    elif r < TRAIN_RATIO + VAL_RATIO:
        split = "val"
    else:
        split = "test"

    # --- Scena ---
    pusta = random.random() < P_PUSTA_SCENA
    n_zadane = 0 if pusta else random.randint(MIN_KLOCKI, MAX_KLOCKI)
    klocki, kolory = postaw_klocki(n_zadane)    # 0 -> pusta scena (tło)
    if not klocki and not pusta:
        skip += 1
        continue

    rozpraszacze = postaw_rozpraszacze(random.randint(0, MAX_DYSTR), klocki)

    # Materiał półki (tekstura lub procedura)
    kolor_ref = kolory[0] if kolory else (0.5,) * 3
    if not ustaw_teksture_polki():
        losuj_material_polki(kolor_ref)

    ustaw_kamere()
    ustaw_dof(klocki)
    ustaw_tlo()
    losuj_swiatla()
    ustaw_compositor()

    # --- Bounding boxy ---
    bpy.context.view_layer.update()
    # każdy widoczny klocek dostaje etykietę (także mały / przycięty kadrem)
    etykiety = [e for e in (bbox_yolo(k) for k in klocki) if e]

    # --- Render i zapis (pusty plik .txt = poprawna ramka tła dla YOLO) ---
    img = os.path.join(BASE_DIR, "images", split, f"img_{frame:04d}.png")
    label = os.path.join(BASE_DIR, "labels", split, f"img_{frame:04d}.txt")

    scene.render.filepath = img
    bpy.ops.render.render(write_still=True)

    with open(label, "w") as f:
        for e in etykiety:
            f.write(f"{CLASS_ID} {e[0]:.6f} {e[1]:.6f} "
                    f"{e[2]:.6f} {e[3]:.6f}\n")

    counts[split] += 1
    frame += 1
    if frame % 100 == 0:
        print(f"[{frame}] train={counts['train']} val={counts['val']} "
              f"test={counts['test']}")

    # --- Sprzątanie ---
    usun_klocki(klocki)
    usun_rozpraszacze(rozpraszacze)
    usun_osierocone_materialy()


# ============================================================
# ZAPIS KONFIGURACJI
# ============================================================
config = {
    "generated": frame - frame0, "skipped": skip,
    "first_frame": frame0,
    **counts,
    "num_samples": NUM_SAMPLES,
    "seed": SEED,
    "cleared_dataset": WYCZYSC_DATASET,
    "resolution": [scene.render.resolution_x, scene.render.resolution_y],
    "class_id": CLASS_ID,
    "engine": scene.render.engine,
    "params": {
        "min_klocki": MIN_KLOCKI, "max_klocki": MAX_KLOCKI,
        "max_dystr": MAX_DYSTR,
        "train_ratio": TRAIN_RATIO, "val_ratio": VAL_RATIO,
        "p_pusta_scena": P_PUSTA_SCENA, "p_brak_dof": P_BRAK_DOF,
        "zakres_xy": ZAKRES_XY, "min_bbox_px": MIN_BBOX_PX,
    },
}
with open(os.path.join(BASE_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"\nGotowe! Frames: {frame - frame0}, skip: {skip}, {counts}")
