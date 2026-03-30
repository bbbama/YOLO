import bpy
import os

# obiekty
obj1 = bpy.data.objects["Cube1"]  # kostka
obj2 = bpy.data.objects["Cube2"]  # półka

# parametry planszy
size = 5
step = 1

# folder zapisu
output_dir = "/Users/bartek/Desktop/Inżynierka/lego_dataset/"
os.makedirs(output_dir, exist_ok=True)

scene = bpy.context.scene

# -----------------------------
# Render Eevee minimalny i szybki
# -----------------------------
scene.render.engine = 'BLENDER_EEVEE'
scene.render.resolution_x = 256
scene.render.resolution_y = 256
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'

# -----------------------------
# Półka stała
# -----------------------------
obj2.animation_data_clear()
obj2.location = (0, 0, -2)

# -----------------------------
# Generowanie datasetu
# -----------------------------
frame = 0
for y in range(size):
    for x in range(size):
        # ustawienie nowej pozycji kostki
        obj1.location = (x*step, y*step, 0)

        # ustawienie sceny na unikalną klatkę
        scene.frame_set(frame)

        # render do pliku
        filename = f"{output_dir}/img_{frame:04d}.png"
        scene.render.filepath = filename
        bpy.ops.render.render(write_still=True)

        frame += 1