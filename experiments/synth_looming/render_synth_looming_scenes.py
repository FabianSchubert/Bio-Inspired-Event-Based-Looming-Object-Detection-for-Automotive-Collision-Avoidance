from src.synth_data.render_looming_scene import render_looming_scene
from itertools import product

import os

VEL_MPS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

OBJECT_FILES = [
    "../../src/synth_data/objects/disc_bright.blend",
    "../../src/synth_data/objects/disc_dark.blend",
    "../../src/synth_data/objects/square_bright.blend",
    "../../src/synth_data/objects/square_dark.blend",
]

OBJECT_FILES = [os.path.join(os.path.dirname(__file__), objf) for objf in OBJECT_FILES]

BACKGROUND_FILES = ["../../src/synth_data/backgrounds/gray_bg.blend", "../../src/synth_data/backgrounds/cloudy_bg.blend"]
BACKGROUND_FILES = [os.path.join(os.path.dirname(__file__), bgf) for bgf in BACKGROUND_FILES]

T_VIDEO = 10.0
FPS = 100

WIDTH, HEIGHT = (304, 240)

CAM_FOV = 45.0  # field of view in deg

NUM_CPU_THREADS = 6
DEVICE = "GPU"
RENDERER = "CYCLES"

base_fold = os.path.join(os.path.dirname(__file__), "../../data/synthetic/")

for vel, obj_f, bg_f in product(VEL_MPS, OBJECT_FILES, BACKGROUND_FILES):
    obj_name = os.path.basename(obj_f).split(".blend")[0]
    bg_name = os.path.basename(bg_f).split(".blend")[0]
    save_fold = os.path.join(base_fold, f"{obj_name}_on_{bg_name}_{vel}_mps/")
    render_looming_scene(
        obj_f,
        bg_f,
        save_fold,
        vel,
        T_VIDEO,
        FPS,
        WIDTH,
        HEIGHT,
        CAM_FOV,
        num_threads=NUM_CPU_THREADS,
        device=DEVICE,
        force_overwrite=True, # don't ask whether to overwrite previously rendered images
        renderer=RENDERER
    )
