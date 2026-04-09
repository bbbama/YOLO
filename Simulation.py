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
obj1 = bpy.data.objects.get("LEGO-2X4-L")
obj2 = bpy.data.objects.get("Cube2")
cam = bpy.data.objects.get("Camera")
light = bpy.data.objects.get("Light")
num_samples = 5
scene = bpy.context.scene
base_dir = "/Users/bartek/Desktop/Inżynierka/lego_dataset/"

# -----------------------------
# Walidacja obiektów
# -----------------------------
assert obj1 is not None, "Nie znaleziono LEGO-2X4-L!"
assert obj2 is not None, "Nie znaleziono Cube2!"
assert cam is not None, "Nie znaleziono Camera!"

# -----------------------------
# Reprodukowalność
# -----------------------------
random.seed(42)

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

# -----------------------------
# Funkcja: unikalne materiały
# -----------------------------
def ensure_unique_material(obj, mat_name):
    """Upewnia się że obiekt ma własny, unikalny materiał."""
    if obj.active_material is None:
        mat = bpy.data.materials.new(name=mat_name)
        mat.node_tree = True
        obj.active_material = mat
    elif obj.active_material.users > 1 or obj.active_material.name != mat_name:
        mat = obj.active_material.copy()
        mat.name = mat_name
        obj.active_material = mat

ensure_unique_material(obj1, "Mat_Klocek")
ensure_unique_material(obj2, "Mat_Polka")

# -----------------------------
# Funkcja: losowe tło świata
# -----------------------------
def randomize_world_background():
    """
    Losuje kolor i intensywność tła sceny (światło otoczenia).
    Zapobiega overfittingowi modelu na stałe tło.
    """
    world = scene.world
    if not world:
        return

    if not world.node_tree:
        world.node_tree = True

    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = (
            random.random(),
            random.random(),
            random.random(),
            1.0
        )
        bg_node.inputs["Strength"].default_value = random.uniform(0.3, 1.5)

# -----------------------------
# Funkcja: losowa pozycja kamery
# -----------------------------
def randomize_camera(cam, target_obj):
    r = random.uniform(8, 14)
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(0.15, 0.55)

    cam.location = mathutils.Vector((
        r * math.sin(phi) * math.cos(theta),
        r * math.sin(phi) * math.sin(theta),
        r * math.cos(phi)
    ))

    direction = target_obj.location - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

# -----------------------------
# Funkcja: losowy kolor klocka
# -----------------------------
def randomize_material_color(obj):
    """Losuje kolor materiału obiektu, zwraca kolor RGB jako tuple."""
    mat = obj.active_material
    if not mat:
        return None

    if mat.users > 1:
        mat = mat.copy()
        obj.active_material = mat

    new_color = (random.random(), random.random(), random.random(), 1.0)

    if mat.node_tree:
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = new_color
        else:
            for node in mat.node_tree.nodes:
                if "Color" in node.inputs:
                    node.inputs["Color"].default_value = new_color
                    break
    else:
        mat.diffuse_color = new_color

    return new_color[:3]

# -----------------------------
# Funkcja: losowy kolor półki
# -----------------------------
def randomize_shelf_color(shelf_obj, avoid_color, min_dist=0.4):
    """
    Losuje kolor półki, gwarantując że będzie wystarczająco różny od klocka.
    avoid_color: tuple RGB klocka
    min_dist: minimalny dystans euklidesowy w przestrzeni RGB (0.0 - 1.73)
    """
    mat = shelf_obj.active_material
    if not mat:
        return

    for _ in range(10):
        new_color = (random.random(), random.random(), random.random(), 1.0)
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(new_color[:3], avoid_color)))
        if dist > min_dist:
            break

    if mat.node_tree:
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = new_color
    else:
        mat.diffuse_color = new_color

# -----------------------------
# Funkcja: YOLO Bbox
# -----------------------------
def get_yolo_bbox(scene, cam, obj):
    bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    co_2d = [bpy_extras.object_utils.world_to_camera_view(scene, cam, c) for c in bbox_corners]

    x_coords = [c.x for c in co_2d]
    y_coords = [1.0 - c.y for c in co_2d]

    min_x, max_x = max(0.0, min(x_coords)), min(1.0, max(x_coords))
    min_y, max_y = max(0.0, min(y_coords)), min(1.0, max(y_coords))

    if min_x == max_x or min_y == max_y:
        return None

    width = max_x - min_x
    height = max_y - min_y
    center_x = min_x + width / 2
    center_y = min_y + height / 2

    return (center_x, center_y, width, height)

# -----------------------------
# Reset półki
# -----------------------------
obj2.animation_data_clear()
obj2.location = (0, 0, -2)

# -----------------------------
# Generowanie datasetu
# -----------------------------
frame = 0
skipped = 0
train_count = 0
val_count = 0

for i in range(num_samples):
    # Losowy podział train/val
    split = "train" if random.random() < 0.8 else "val"

    # 1. Losowa pozycja klocka
    obj1.location = mathutils.Vector((
        random.uniform(-2.0, 2.0),
        random.uniform(-2.0, 2.0),
        -1
    ))

    # 2. Losowa rotacja Z
    obj1.rotation_euler = (0, 0, random.uniform(0, 2 * math.pi))

    # 3. Losowy kolor klocka, a następnie półki (inny niż klocek)
    lego_color = randomize_material_color(obj1)
    if lego_color:
        randomize_shelf_color(obj2, lego_color)

    # 4. Losowa pozycja kamery
    randomize_camera(cam, obj1)

    # 5. Losowe tło
    randomize_world_background()

    # 6. Losowe światło
    if light:
        light.location = (
            random.uniform(-5, 5),
            random.uniform(-5, 5),
            random.uniform(2, 6)
        )
        light.data.energy = random.uniform(100, 1000)

    # 7. Aktualizacja widoku
    bpy.context.view_layer.update()

    # 8. Wyliczenie bounding box
    bbox = get_yolo_bbox(scene, cam, obj1)

    if bbox and bbox[2] > 0.01 and bbox[3] > 0.01:
        img_path = os.path.join(base_dir, "images", split, f"img_{frame:04d}.png")
        label_path = os.path.join(base_dir, "labels", split, f"img_{frame:04d}.txt")

        scene.render.filepath = img_path
        bpy.ops.render.render(write_still=True)

        with open(label_path, "w") as f:
            f.write(f"{CLASS_ID} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

        if split == "train":
            train_count += 1
        else:
            val_count += 1

        frame += 1

        if frame % 10 == 0:
            print(f"Postęp: {frame}/{num_samples} | train: {train_count} | val: {val_count}")
    else:
        skipped += 1
        print(f"Pominięto klatkę {i} (obiekt poza kadrem lub za mały)")

# -----------------------------
# Zapis konfiguracji
# -----------------------------
config = {
    "num_samples": num_samples,
    "generated": frame,
    "skipped": skipped,
    "train": train_count,
    "val": val_count,
    "train_ratio": 0.8,
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

#Przetosowac tryb batchowy
#Ramka na klocek do labelowania