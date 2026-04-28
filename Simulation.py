import bpy
import bpy_extras
import os
import random
import math
import mathutils
import json

# -----------------------------
# Konfiguracja
# -----------------------------
CLASS_ID = 0
obj_wzor = bpy.data.objects.get("LEGO-2X4-L")  # Obiekt-wzór, ukryty, tylko do kopiowania
obj_polka = bpy.data.objects.get("Cube2")
cam = bpy.data.objects.get("Camera")
light = bpy.data.objects.get("Light")
num_samples = 5
scene = bpy.context.scene
base_dir = "/Users/bartek/Desktop/Inżynierka/lego_dataset/"

# Ile klocków maksymalnie na jednym zdjęciu
MIN_KLOCKOW = 1
MAX_KLOCKOW = 10

# -----------------------------
# Walidacja obiektów
# -----------------------------
assert obj_wzor is not None, "Nie znaleziono LEGO-2X4-L!"
assert obj_polka is not None, "Nie znaleziono Cube2!"
assert cam is not None, "Nie znaleziono Camera!"

# Ukrywamy obiekt-wzór – służy tylko jako źródło do kopiowania
obj_wzor.hide_render = True
obj_wzor.hide_viewport = True

# -----------------------------
# Reprodukowalność
# -----------------------------
# random.seed(42)  # odkomentuj tylko jeśli potrzebujesz reprodukowalności

# -----------------------------
# Tworzenie struktury katalogów
# -----------------------------
for split in ["train", "val"]:
    os.makedirs(os.path.join(base_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "labels", split), exist_ok=True)

# -----------------------------
# Zapis data.yaml dla YOLO
# -----------------------------
yaml_content = (
    f"path: {base_dir}\n"
    f"train: images/train\n"
    f"val: images/val\n"
    f"\n"
    f"nc: 1\n"
    f"names: ['lego_2x4']\n"
)
with open(os.path.join(base_dir, "data.yaml"), "w") as f:
    f.write(yaml_content)

# -----------------------------
# Render Eevee
# -----------------------------
scene.render.engine = 'BLENDER_EEVEE'
scene.render.resolution_x = 640
scene.render.resolution_y = 640
scene.render.resolution_percentage = 100
scene.render.filter_size = 2.0
scene.eevee.taa_render_samples = 64
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGB'
scene.render.image_settings.compression = 0

# ================================================================
# FUNKCJE POMOCNICZE
# ================================================================

def daj_nowy_material(obj, nazwa):
    """Tworzy nowy unikalny materiał i przypisuje go do obiektu."""
    mat = bpy.data.materials.new(name=nazwa)
    mat.use_nodes = True
    obj.active_material = mat
    return mat


def pobierz_bsdf(obj):
    """Zwraca węzeł Principled BSDF materiału obiektu lub None."""
    mat = obj.active_material
    if not mat or not mat.node_tree:
        return None
    return mat.node_tree.nodes.get("Principled BSDF")


def losuj_material_klocka(obj):
    """Losuje kolor, szorstkość i metaliczność klocka. Zwraca kolor RGB."""
    bsdf = pobierz_bsdf(obj)
    if not bsdf:
        return None

    kolor = (random.random(), random.random(), random.random(), 1.0)
    bsdf.inputs["Base Color"].default_value = kolor
    bsdf.inputs["Roughness"].default_value = random.uniform(0.05, 0.5)
    bsdf.inputs["Metallic"].default_value = random.uniform(0.0, 0.1)

    return kolor[:3]


def losuj_material_polki(obj_polka, unikaj_koloru, min_dystans=0.4):
    """
    Losuje kolor półki z szumem proceduralnym,
    dbając o kontrast z podanym kolorem klocka.
    """
    mat = obj_polka.active_material
    if not mat or not mat.node_tree:
        return

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if not bsdf:
        return

    # Odepnij stare połączenia do Base Color
    for link in list(bsdf.inputs["Base Color"].links):
        links.remove(link)

    # Losuj kolor podstawowy z minimalnym dystansem od klocka
    for _ in range(10):
        c1 = (random.random(), random.random(), random.random(), 1.0)
        dystans = math.sqrt(sum((a - b) ** 2 for a, b in zip(c1[:3], unikaj_koloru)))
        if dystans > min_dystans:
            break

    c2 = (random.random(), random.random(), random.random(), 1.0)

    # Węzeł szumu proceduralnego
    tex_node = nodes.get("Noise_Tex") or nodes.new("ShaderNodeTexNoise")
    tex_node.name = "Noise_Tex"
    tex_node.inputs["Scale"].default_value = random.uniform(2.0, 50.0)
    tex_node.inputs["Detail"].default_value = 15.0

    # Węzeł mieszania kolorów
    mix_node = nodes.get("Mix_Col") or nodes.new("ShaderNodeMixRGB")
    mix_node.name = "Mix_Col"
    mix_node.inputs["Color1"].default_value = c1
    mix_node.inputs["Color2"].default_value = c2
    mix_node.inputs["Fac"].default_value = random.uniform(0.1, 0.6)

    links.new(tex_node.outputs["Fac"], mix_node.inputs["Fac"])
    links.new(mix_node.outputs["Color"], bsdf.inputs["Base Color"])

    bsdf.inputs["Roughness"].default_value = random.uniform(0.1, 1.0)
    bsdf.inputs["Metallic"].default_value = random.uniform(0.0, 0.4)


def losuj_tlo_swiata():
    """Losuje kolor i intensywność tła sceny."""
    world = scene.world
    if not world or not world.node_tree:
        return

    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = (
            random.random(), random.random(), random.random(), 1.0
        )
        bg_node.inputs["Strength"].default_value = random.uniform(0.3, 1.5)


def losuj_swiatlo(light):
    """Losuje pozycję, moc i barwę światła."""
    if not light:
        return
    light.location = (
        random.uniform(-5, 5),
        random.uniform(-5, 5),
        random.uniform(2, 6)
    )
    light.data.energy = random.uniform(100, 1000)
    light.data.color = (1.0, random.uniform(0.8, 1.0), random.uniform(0.7, 1.0))


def losuj_kamere(cam, cel):
    """Ustawia kamerę w losowej pozycji sferycznej skierowanej na obiekt cel."""
    r = random.uniform(12, 18)
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(0.15, 0.55)

    cam.location = mathutils.Vector((
        r * math.sin(phi) * math.cos(theta),
        r * math.sin(phi) * math.sin(theta),
        r * math.cos(phi)
    ))

    kierunek = cel.location - cam.location
    cam.rotation_euler = kierunek.to_track_quat('-Z', 'Y').to_euler()


def czy_nachodza(obj1, obj2, margines=0.2):
    """
    Sprawdza kolizję przez porównanie okręgów okalających oba klocki.
    Promień okręgu = połowa przekątnej klocka, więc obejmuje go w pełni
    niezależnie od obrotu. Używamy obj.dimensions i obj.location zamiast
    matrix_world, bo są zawsze aktualne bez potrzeby update().
    """
    def promien(obj):
        w = obj.dimensions
        return math.sqrt((w.x / 2) ** 2 + (w.y / 2) ** 2)

    dystans = math.sqrt(
        (obj1.location.x - obj2.location.x) ** 2 +
        (obj1.location.y - obj2.location.y) ** 2
    )
    return dystans < (promien(obj1) + promien(obj2) + margines)


def oblicz_bbox_yolo(scene, cam, obj):
    """Oblicza bounding box obiektu w formacie YOLO (cx, cy, w, h) w zakresie [0,1]."""
    bpy.context.view_layer.update()
    corners = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    coords_2d = [bpy_extras.object_utils.world_to_camera_view(scene, cam, c) for c in corners]

    x_points = [c.x for c in coords_2d]
    y_points = [1.0 - c.y for c in coords_2d]

    min_x, max_x = max(0.0, min(x_points)), min(1.0, max(x_points))
    min_y, max_y = max(0.0, min(y_points)), min(1.0, max(y_points))

    if min_x >= max_x or min_y >= max_y:
        return None

    w = max_x - min_x
    h = max_y - min_y
    return (min_x + w / 2, min_y + h / 2, w, h)


def stworz_klocki_na_scenie(obj_wzor, liczba_klockow):
    """
    Klonuje obj_wzor wielokrotnie, rozmieszcza klocki bez nakładania się.
    Zwraca listę stworzonych obiektów.
    """
    stworzone = []

    for j in range(liczba_klockow):
        nowy = obj_wzor.copy()
        nowy.data = obj_wzor.data.copy()
        bpy.context.collection.objects.link(nowy)

        znaleziono = False
        for _ in range(100):
            nowy.location = mathutils.Vector((
                random.uniform(-2, 2),
                random.uniform(-2, 2),
                -1
            ))
            nowy.rotation_euler = (0, 0, random.uniform(0, 2 * math.pi))
            bpy.context.view_layer.update()

            nachodzi = any(czy_nachodza(nowy, s) for s in stworzone)

            if not nachodzi:
                znaleziono = True
                break

        if znaleziono:
            nowy.hide_render = False
            nowy.hide_viewport = False
            # Każdy klocek dostaje własny materiał
            daj_nowy_material(nowy, f"Mat_Klocek_{j}")
            losuj_material_klocka(nowy)
            stworzone.append(nowy)
        else:
            # Nie znaleziono miejsca – usuwamy kopię
            mesh = nowy.data
            bpy.data.objects.remove(nowy, do_unlink=True)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)

    return stworzone


def usun_klocki(lista_klockow):
    """Usuwa wszystkie tymczasowe obiekty klocków ze sceny."""
    for klocek in lista_klockow:
        mesh = klocek.data
        bpy.data.objects.remove(klocek, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


# ================================================================
# Przygotowanie stałych elementów sceny
# ================================================================

# Półka ma własny materiał (raz, nie zmieniamy struktury węzłów przy każdej klatce)
if obj_polka.active_material is None or obj_polka.active_material.users > 1:
    daj_nowy_material(obj_polka, "Mat_Polka")

obj_polka.location = (0, 0, -2)

# ================================================================
# Główna pętla generowania datasetu
# ================================================================
frame = 0
skipped = 0
train_count = 0
val_count = 0

for i in range(num_samples):
    split = "train" if random.random() < 0.8 else "val"

    # 1. Losuj liczbę klocków na tej klatce
    liczba_klockow = random.randint(MIN_KLOCKOW, MAX_KLOCKOW)

    # 2. Postaw klocki na scenie bez nakładania się
    klocki = stworz_klocki_na_scenie(obj_wzor, liczba_klockow)

    if not klocki:
        print(f"Pominięto klatkę {i} – nie udało się postawić żadnego klocka.")
        skipped += 1
        continue

    # 3. Wylosuj kolor półki kontrastujący z pierwszym klockiem
    bsdf_ref = pobierz_bsdf(klocki[0])
    kolor_ref = tuple(bsdf_ref.inputs["Base Color"].default_value[:3]) if bsdf_ref else (0.5, 0.5, 0.5)
    losuj_material_polki(obj_polka, kolor_ref)

    # 4. Ustaw kamerę, tło i światło
    losuj_kamere(cam, obj_polka)
    losuj_tlo_swiata()
    losuj_swiatlo(light)

    # 7. Aktualizacja i wyliczenie bounding boxów
    bpy.context.view_layer.update()

    yolo_labels = []
    for klocek in klocki:
        bbox = oblicz_bbox_yolo(scene, cam, klocek)
        if bbox and bbox[2] > 0.01 and bbox[3] > 0.01:
            yolo_labels.append(bbox)

    # 8. Render i zapis (tylko jeśli jest co labelować)
    if yolo_labels:
        img_path = os.path.join(base_dir, "images", split, f"img_{frame:04d}.png")
        label_path = os.path.join(base_dir, "labels", split, f"img_{frame:04d}.txt")

        scene.render.filepath = img_path
        bpy.ops.render.render(write_still=True)

        with open(label_path, "w") as f:
            for lb in yolo_labels:
                f.write(f"{CLASS_ID} {lb[0]:.6f} {lb[1]:.6f} {lb[2]:.6f} {lb[3]:.6f}\n")

        if split == "train":
            train_count += 1
        else:
            val_count += 1

        frame += 1

        print(f"Zdjęcie {frame}: Klocków: {len(klocki)}")
    else:
        skipped += 1
        print(f"Pominięto klatkę {i} (wszystkie klocki poza kadrem lub za małe)")

    # 9. Sprzątanie – usuwamy tymczasowe klocki
    usun_klocki(klocki)

# ================================================================
# Zapis konfiguracji
# ================================================================
config = {
    "num_samples": num_samples,
    "generated": frame,
    "skipped": skipped,
    "train": train_count,
    "val": val_count,
    "train_ratio": 0.8,
    "min_klockow": MIN_KLOCKOW,
    "max_klockow": MAX_KLOCKOW,
    "resolution": [scene.render.resolution_x, scene.render.resolution_y],
    "class_id": CLASS_ID,
    "engine": scene.render.engine,
}
with open(os.path.join(base_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"\nZakończono!")
print(f"  Wygenerowano: {frame} klatek")
print(f"  Train:        {train_count}")
print(f"  Val:          {val_count}")
print(f"  Pominięto:    {skipped}")