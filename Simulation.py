import bpy
import bpy_extras
import os
import random
import math
import mathutils
import json
import glob

# ============================================================
# KONFIGURACJA
# ============================================================
CLASS_ID      = 0
NUM_SAMPLES   = 1000
MIN_KLOCKI    = 1
MAX_KLOCKI    = 10
MAX_DYSTR     = 8        # rozpraszacze (distractor objects)

TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
# reszta idzie do test

NUM_LIGHTS_MIN = 1
NUM_LIGHTS_MAX = 3

USE_TEX_BG    = True     # tekstura tła sceny
USE_TEX_SHELF = True     # tekstura półki

script_dir    = os.path.dirname(os.path.abspath(__file__))
BASE_DIR      = os.path.join(script_dir, "lego_dataset")
TEX_BG_DIR    = os.path.join(script_dir, "textures", "backgrounds")
TEX_SHELF_DIR = os.path.join(script_dir, "textures", "shelf")

# ============================================================
# POBRANIE OBIEKTÓW ZE SCENY
# ============================================================
obj_wzor  = bpy.data.objects["LEGO-2X4-L"]   # wzór klocka - tylko do kopiowania
obj_polka = bpy.data.objects["Cube2"]          # półka
cam       = bpy.data.objects["Camera"]

obj_wzor.hide_render   = True
obj_wzor.hide_viewport = True

scene = bpy.context.scene

# ============================================================
# USTAWIENIA RENDERERA
# ============================================================
scene.render.engine                       = 'BLENDER_EEVEE'
scene.render.resolution_x                 = 640
scene.render.resolution_y                 = 640
scene.render.resolution_percentage        = 100
scene.render.filter_size                  = 2.0
scene.eevee.taa_render_samples            = 64
scene.render.image_settings.file_format   = 'PNG'
scene.render.image_settings.color_mode    = 'RGB'
scene.render.image_settings.compression   = 0
scene.render.use_motion_blur              = True

# ============================================================
# TWORZENIE KATALOGÓW I PLIKU YAML
# ============================================================
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(BASE_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "labels", split), exist_ok=True)

with open(os.path.join(BASE_DIR, "data.yaml"), "w") as f:
    f.write(
        f"path: {BASE_DIR}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"\nnc: 1\nnames: ['lego_2x4']\n"
    )

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

bg_textures    = wczytaj_tekstury(TEX_BG_DIR)
shelf_textures = wczytaj_tekstury(TEX_SHELF_DIR)
print(f"Tekstury tła: {len(bg_textures)}, półki: {len(shelf_textures)}")

# ============================================================
# POMOCNIK: PRINCIPLED BSDF
# ============================================================
def bsdf(obj):
    """Zwraca węzeł Principled BSDF obiektu (lub None)."""
    mat = obj.active_material
    return mat.node_tree.nodes.get("Principled BSDF") if mat and mat.node_tree else None

def nowy_material(obj, nazwa):
    """Przypisuje świeży materiał do obiektu i zwraca węzeł BSDF."""
    mat = bpy.data.materials.new(nazwa)
    obj.active_material = mat
    return mat.node_tree.nodes.get("Principled BSDF")

# ============================================================
# MATERIAŁY
# ============================================================
def losuj_material_klocka(obj, idx):
    node = nowy_material(obj, f"Mat_Klocek_{idx}")
    if not node:
        return (0.5, 0.5, 0.5)
    kolor = (random.random(), random.random(), random.random(), 1.0)
    node.inputs["Base Color"].default_value  = kolor
    node.inputs["Roughness"].default_value   = random.uniform(0.05, 0.5)
    node.inputs["Metallic"].default_value    = random.uniform(0.0, 0.1)
    return kolor[:3]

def losuj_material_polki(kolor_klocka):
    """Proceduralny szum na półce; kontrast z kolorem klocka."""
    mat = obj_polka.active_material
    if not mat or not mat.node_tree:
        return
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    node  = nodes.get("Principled BSDF")
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
    noise.inputs["Scale"].default_value  = random.uniform(2.0, 50.0)
    noise.inputs["Detail"].default_value = 15.0

    mix = nodes.get("Mix_Col") or nodes.new("ShaderNodeMixRGB")
    mix.name = "Mix_Col"
    mix.inputs["Color1"].default_value = c1
    mix.inputs["Color2"].default_value = c2
    mix.inputs["Fac"].default_value    = random.uniform(0.1, 0.6)

    links.new(noise.outputs["Fac"], mix.inputs["Fac"])
    links.new(mix.outputs["Color"], node.inputs["Base Color"])

    node.inputs["Roughness"].default_value = random.uniform(0.1, 1.0)
    node.inputs["Metallic"].default_value  = random.uniform(0.0, 0.4)

def ustaw_teksture_polki():
    """Nakłada losową teksturę na półkę; zwraca True jeśli się udało."""
    if not USE_TEX_SHELF or not shelf_textures:
        return False
    mat  = obj_polka.active_material
    node = mat.node_tree.nodes.get("Principled BSDF") if mat and mat.node_tree else None
    if not node:
        return False

    for lnk in list(node.inputs["Base Color"].links):
        mat.node_tree.links.remove(lnk)

    tex   = mat.node_tree.nodes.get("Shelf_Tex") or mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.name = "Shelf_Tex"
    coord = mat.node_tree.nodes.get("Tex_Coord") or mat.node_tree.nodes.new("ShaderNodeTexCoord")
    coord.name = "Tex_Coord"

    try:
        tex.image = bpy.data.images.load(random.choice(shelf_textures), check_existing=True)
        mat.node_tree.links.new(coord.outputs["UV"], tex.inputs["Vector"])
        mat.node_tree.links.new(tex.outputs["Color"], node.inputs["Base Color"])
        return True
    except Exception:
        return False

def ustaw_tlo():
    """Losuje tło: tekstura HDRI lub kolor proceduralny."""
    world  = scene.world
    bg     = world.node_tree.nodes.get("Background") if world and world.node_tree else None
    if not bg:
        return

    if USE_TEX_BG and bg_textures:
        tex = world.node_tree.nodes.get("Bg_Tex") or world.node_tree.nodes.new("ShaderNodeTexEnvironment")
        tex.name = "Bg_Tex"
        try:
            tex.image = bpy.data.images.load(random.choice(bg_textures), check_existing=True)
            world.node_tree.links.new(tex.outputs["Color"], bg.inputs["Color"])
            bg.inputs["Strength"].default_value = random.uniform(0.5, 2.0)
            return
        except Exception:
            pass

    # fallback – losowy kolor
    bg.inputs["Color"].default_value    = (random.random(), random.random(), random.random(), 1.0)
    bg.inputs["Strength"].default_value = random.uniform(0.3, 1.5)

# ============================================================
# KAMERA
# ============================================================
def ustaw_kamere():
    """Losowa pozycja sferyczna, skierowana na półkę + losowa ogniskowa."""
    r     = random.uniform(8, 22)
    theta = random.uniform(0, 2 * math.pi)
    phi   = random.uniform(0.05, 0.85)
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
    # Usuń stare
    for o in [o for o in bpy.data.objects if o.type == 'LIGHT']:
        bpy.data.objects.remove(o, do_unlink=True)

    for i in range(random.randint(NUM_LIGHTS_MIN, NUM_LIGHTS_MAX)):
        typ = random.choice(['POINT', 'SUN', 'AREA'])
        bpy.ops.object.light_add(type=typ)
        lt = bpy.context.active_object
        lt.name = f"Light_{i}"

        if typ == 'SUN':
            lt.location = (0, 0, 10)
            lt.rotation_euler = (random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(0, 2*math.pi))
            lt.data.energy = random.uniform(2, 8)
        else:
            lt.location = (random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(2, 8))
            lt.data.energy = random.uniform(200, 1500)
            if typ == 'AREA':
                lt.data.size = random.uniform(1, 5)

        lt.data.color = (random.uniform(0.8, 1.0), random.uniform(0.8, 1.0), random.uniform(0.7, 1.0))

# ============================================================
# SPRAWDZANIE KOLIZJI
# ============================================================
def nachodza(a, b, margines=0.2):
    """True jeśli okręgi okalające obiekty a i b zachodzą na siebie."""
    def r(o):
        d = o.dimensions
        return math.sqrt((d.x/2)**2 + (d.y/2)**2)
    dist = math.sqrt((a.location.x - b.location.x)**2 + (a.location.y - b.location.y)**2)
    return dist < r(a) + r(b) + margines

# ============================================================
# KLOCKI
# ============================================================
def postaw_klocki(n):
    """Klonuje wzór n razy, rozmieszcza bez kolizji. Zwraca listę obiektów."""
    klocki = []
    for idx in range(n):
        klon = obj_wzor.copy()
        klon.data = obj_wzor.data.copy()
        bpy.context.collection.objects.link(klon)
        klon.hide_render = klon.hide_viewport = False

        for _ in range(100):
            klon.location      = mathutils.Vector((random.uniform(-2, 2), random.uniform(-2, 2), -1))
            klon.rotation_euler = (0, 0, random.uniform(0, 2*math.pi))
            bpy.context.view_layer.update()
            if not any(nachodza(klon, k) for k in klocki):
                break
        else:
            # Brak miejsca – usuń klon
            mesh = klon.data
            bpy.data.objects.remove(klon, do_unlink=True)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
            continue

        losuj_material_klocka(klon, idx)
        klocki.append(klon)
    return klocki

def usun_klocki(klocki):
    for k in klocki:
        mesh = k.data
        bpy.data.objects.remove(k, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)

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
            'CUBE':   'primitive_cube_add',
            'TORUS':  'primitive_torus_add',
            'CONE':   'primitive_cone_add',
        }[shape])()

        obj = bpy.context.active_object
        obj.name  = f"Dystraktor_{i}"
        obj.scale = mathutils.Vector((random.uniform(0.1, 0.4),) * 3)

        for _ in range(50):
            obj.location = mathutils.Vector((random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(-0.5, 1)))
            bpy.context.view_layer.update()
            if not any(nachodza(obj, k) for k in klocki):
                break
        else:
            bpy.data.objects.remove(obj, do_unlink=True)
            continue

        node = nowy_material(obj, f"Mat_Dyst_{i}")
        if node:
            node.inputs["Base Color"].default_value = (random.random(), random.random(), random.random(), 1.0)
            node.inputs["Roughness"].default_value  = random.uniform(0.2, 0.9)

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
    """Ustawia głębię ostrości na losowy klocek."""
    if not klocki:
        return
    cam.data.dof.use_dof = True
    cam.data.dof.aperture_fstop = random.uniform(1.4, 8.0)
    cam.data.dof.focus_object = random.choice(klocki)

def ustaw_compositor():
    """Symulacja efektów optycznych bez Compositora - tylko ogniskowa kamery."""
    # Compositor API jest niestabilne w Blenderze 5.x - pomijamy
    # Efekt lens distortion zastępuje losowa ogniskowa w ustaw_kamere()
    pass

# ============================================================
# BOUNDING BOX YOLO
# ============================================================
def bbox_yolo(obj):
    """Zwraca (cx, cy, w, h) w [0,1] lub None jeśli obiekt poza kadrem (min 20% widoczności)."""
    bpy.context.view_layer.update()
    corners  = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    pts      = [bpy_extras.object_utils.world_to_camera_view(scene, cam, c) for c in corners]
    xs       = [p.x       for p in pts]
    ys       = [1.0 - p.y for p in pts]

    orig_w = max(xs) - min(xs)
    orig_h = max(ys) - min(ys)

    x0, x1   = max(0.0, min(xs)), min(1.0, max(xs))
    y0, y1   = max(0.0, min(ys)), min(1.0, max(ys))

    if x0 >= x1 or y0 >= y1:
        return None

    w, h = x1 - x0, y1 - y0

    # Akceptuj tylko jeśli min 20% wymiaru jest w kadrze
    if w < 0.2 * orig_w or h < 0.2 * orig_h:
        return None

    return (x0 + w/2, y0 + h/2, w, h)

# ============================================================
# GŁÓWNA PĘTLA
# ============================================================

# Upewnij się, że półka ma własny materiał
if obj_polka.active_material is None:
    nowy_material(obj_polka, "Mat_Polka")
obj_polka.location = (0, 0, -2)

counts = {"train": 0, "val": 0, "test": 0}
frame  = 0
skip   = 0

for i in range(NUM_SAMPLES):
    # Wylosuj split
    r = random.random()
    split = "train" if r < TRAIN_RATIO else "val" if r < TRAIN_RATIO + VAL_RATIO else "test"

    # --- Scena ---
    klocki      = postaw_klocki(random.randint(MIN_KLOCKI, MAX_KLOCKI))
    if not klocki:
        skip += 1
        continue

    rozpraszacze = postaw_rozpraszacze(random.randint(0, MAX_DYSTR), klocki)

    # Materiał półki (tekstura lub procedura)
    kolor_ref = bsdf(klocki[0]).inputs["Base Color"].default_value[:3] if bsdf(klocki[0]) else (0.5,)*3
    if not ustaw_teksture_polki():
        losuj_material_polki(kolor_ref)

    ustaw_kamere()
    ustaw_dof(klocki)
    ustaw_tlo()
    losuj_swiatla()
    ustaw_compositor()

    # Losowy motion blur shutter
    scene.render.motion_blur_shutter = random.uniform(0.0, 0.4)

    # --- Bounding boxy ---
    bpy.context.view_layer.update()
    etykiety = [bbox_yolo(k) for k in klocki]
    etykiety  = [e for e in etykiety if e and e[2] > 0.05 and e[3] > 0.05]

    # --- Render i zapis ---
    if etykiety:
        img   = os.path.join(BASE_DIR, "images", split, f"img_{frame:04d}.png")
        label = os.path.join(BASE_DIR, "labels", split, f"img_{frame:04d}.txt")

        scene.render.filepath = img
        bpy.ops.render.render(write_still=True)

        with open(label, "w") as f:
            for e in etykiety:
                f.write(f"{CLASS_ID} {e[0]:.6f} {e[1]:.6f} {e[2]:.6f} {e[3]:.6f}\n")

        counts[split] += 1
        frame += 1
        if frame % 100 == 0:
            print(f"[{frame}] train={counts['train']} val={counts['val']} test={counts['test']}")
    else:
        skip += 1

    # --- Sprzątanie ---
    usun_klocki(klocki)
    usun_rozpraszacze(rozpraszacze)

# ============================================================
# ZAPIS KONFIGURACJI
# ============================================================
config = {
    "generated": frame, "skipped": skip,
    **counts,
    "num_samples": NUM_SAMPLES,
    "resolution": [640, 640],
    "class_id": CLASS_ID,
    "engine": scene.render.engine,
}
with open(os.path.join(BASE_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"\nGotowe! Frames: {frame}, skip: {skip}, {counts}")