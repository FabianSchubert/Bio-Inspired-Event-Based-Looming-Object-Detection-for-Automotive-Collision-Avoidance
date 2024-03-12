from turtle import back
import bpy
import numpy as np

from enum import Enum

import os
import sys

import json

device_mode = Enum("device_mode", ["CPU", "GPU"])


def set_up_looming_scene(
    loom_obj_file: str,
    bg_file: str,
    t_run: float,
    fps: int,
    vel: float,
    width: int,
    height: int,
    render_output_dir: str,
    cam_fov_deg: float = 45.0,
    num_threads: int = -1,  # number of threads to be used by the renderer. -1 means automatic,
    device: str = "CPU",
    renderer: str = "CYCLES",
):
    if not render_output_dir.endswith("/"):
        render_output_dir += "/"

    frame_end = int(t_run * fps)

    d_init = vel * t_run

    cam_fov = cam_fov_deg * 2.0 * np.pi / 360.0

    #### clear the scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False, confirm=False)
    ####

    #### set up scene
    scene = bpy.data.scenes["Scene"]

    scene.frame_start = 1
    scene.frame_end = frame_end

    scene.render.fps = fps

    scene.render.resolution_x = width
    scene.render.resolution_y = height

    scene.render.filepath = render_output_dir
    scene.render.image_settings.file_format = "PNG"
    scene.render.engine = renderer
    scene.cycles.device = device_mode[device].name

    scene.cycles.samples = 10  # less means more grain and stepped edges.

    num_threads = int(num_threads)
    if num_threads < -1 or num_threads == 0:
        print(
            "Warning: Invalid Argument to number of threads: should be > 1 or -1 (automatic)"
        )
        num_threads = -1
    if num_threads == -1:
        scene.render.threads_mode = "AUTO"
    else:
        scene.render.threads_mode = "FIXED"
        scene.render.threads = num_threads
    ####

    #### attach object
    path_to_blendfile = loom_obj_file

    obj_name = "Loom_Obj"
    obj_dir = os.path.join(path_to_blendfile, "Object/")
    path_to_obj = os.path.join(obj_dir, obj_name)
    print(path_to_obj)

    bpy.ops.wm.append(filepath=path_to_obj, directory=obj_dir, filename=obj_name)

    loom_obj = bpy.data.objects["Loom_Obj"]
    ####

    #### attach World (for background)
    # remove old ones
    for wrld in bpy.data.worlds:
        bpy.data.worlds.remove(wrld)

    path_to_blendfile = bg_file

    obj_name = "World"
    obj_dir = os.path.join(path_to_blendfile, "World/")
    path_to_obj = os.path.join(obj_dir, obj_name)

    bpy.ops.wm.append(filepath=path_to_obj, directory=obj_dir, filename=obj_name)

    bg_world = bpy.data.worlds[0]

    scene.world = bg_world
    ####

    #### animate looming object
    loom_obj.keyframe_insert(data_path="location", frame=frame_end)

    loom_obj.location[1] = d_init
    loom_obj.keyframe_insert(data_path="location", frame=1)

    loom_action = bpy.data.actions.get(loom_obj.animation_data.action.name)

    for fcurve in loom_action.fcurves.values():
        for pt in fcurve.keyframe_points:
            pt.interpolation = "LINEAR"
    ####

    #### set up camera
    bpy.ops.object.camera_add(location=(0, 0, 0), rotation=(np.pi / 2.0, 0, 0))
    camera = bpy.data.objects["Camera"]

    camera.data.angle = cam_fov

    scene.camera = camera
    ####

    return scene, loom_obj, camera, bg_world


def render_looming_scene(
    loom_object_file: str,
    background_file: str,
    save_fold: str,
    vel: float | int,
    t_video: float | int,
    fps: int,
    width: int,
    height: int,
    cam_fov_deg: float = 45.0,
    num_threads: int = -1,  # number of threads to be used by the renderer. -1 means automatic
    device: str = "CPU",
    force_overwrite: bool = False,
    renderer: str = "CYCLES",
) -> None:
    fps = int(fps)
    n_frames = int(t_video * fps)
    dist_init = vel * t_video

    metadat = {
        "velocity_mps": vel,
        "t_sec": t_video,
        "fps": fps,
        "n_frames": n_frames,
        "d_init_m": dist_init,
    }

    timestamps = np.arange(n_frames) / fps

    frame_fold = os.path.join(save_fold, "frames/")

    if not os.path.exists(frame_fold):
        os.makedirs(frame_fold)

    files_frame_fold = os.listdir(frame_fold)

    if files_frame_fold:
        if force_overwrite:
            print("Info: Overwriting files in render output folder.")
        else:
            while True:
                resp = input(
                    "Frame folder not empty! Continue overwriting? [y/n] "
                ).lower()
                if resp in ["y", "n"]:
                    if resp == "y":
                        break
                    else:
                        sys.exit()

    for fl in files_frame_fold:
        os.remove(os.path.join(frame_fold, fl))

    with open(os.path.join(save_fold, "metadat.json"), "w") as f:
        json.dump(metadat, f)

    with open(os.path.join(save_fold, "timestamps.txt"), "w") as f:
        f.write("\n".join(timestamps.astype(str)))

    scene, loom_obj, camera, bg_world = set_up_looming_scene(
        loom_object_file,
        background_file,
        t_video,
        fps,
        vel,
        width,
        height,
        frame_fold,
        cam_fov_deg,
        num_threads=num_threads,
        device=device,
        renderer=renderer,
    )

    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    VELOCITY_MPS = 2.5
    T_VIDEO_SEC = 10.0
    FPS = int(100.0)

    base_fold = os.path.dirname(__file__)

    render_looming_scene(
        os.path.join(base_fold, "objects/circle_bright.blend"),
        os.path.join(base_fold, "backgrounds/gray_bg.blend"),
        os.path.join(base_fold, "test_render/"),
        VELOCITY_MPS,
        T_VIDEO_SEC,
        FPS,
        304,
        240,
    )
