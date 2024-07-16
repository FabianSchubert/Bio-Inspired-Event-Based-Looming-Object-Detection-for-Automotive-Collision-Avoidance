"""
carla simulator utils
"""

import carla

from carla import ColorConverter as cc

import weakref

import numpy as np

import pygame

import os

import math

import datetime

from src.config import EVENTS_DTYPE

def calc_projected_box_extent(inv_cam_matrix, box_world_vertices):
    x0, y0, x1, y1 = np.inf, np.inf, -np.inf, -np.inf
    rot_mat = np.array(inv_cam_matrix)[:3, :3]
    for vertex in box_world_vertices:
        vertex = np.array([vertex.x, vertex.y, vertex.z])
        # rotate vertex into camera frame
        vertex = np.dot(rot_mat, vertex)
        # transform from UE4 to "standard" right-handed coordinates:
        # in the camera space of ue4:
        # forward is x, right is y, up is z
        # in the "normal" camera space:
        # right is x, down is y, forward is z
        vertex = np.array([vertex[1], -vertex[2], vertex[0]])
        # update the outer bounds
        x0 = min(x0, vertex[0])
        y0 = min(y0, vertex[1])
        x1 = max(x1, vertex[0])
        y1 = max(y1, vertex[1])
    
    return (x1 - x0), (y1 - y0)

def draw_image(surface, image, blend=False, scale=None):
    if isinstance(image, carla.libcarla.Image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
    else:
        array = image.copy()
    scale = (image.width, image.height) if scale is None else scale
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    image_surface = pygame.transform.scale(image_surface, scale)
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))

def pol_evt_img_to_rgb(evt_img):
    rgb_img = np.zeros((evt_img.shape[0], evt_img.shape[1], 3), dtype=np.uint8)
    rgb_img[:, :, 0] = (evt_img == -1) * 255
    rgb_img[:, :, 1] = (evt_img == 1) * 255

    return rgb_img

def convert_events(events):
    evt_arr = np.frombuffer(
        events.raw_data,
        dtype=np.dtype(
            [
                ("x", np.uint16),
                ("y", np.uint16),
                ("t", np.int64),
                ("p", bool),
            ]
        ),
    ).copy()
    # evt_arr['p'] = evt_arr['p'].astype(int) * 2 - 1
    return evt_arr


def downsample_events(events_array, width, height, clip=1):
    idx_x = events_array["x"].astype(int)
    idx_y = events_array["y"].astype(int)
    idx_flat = idx_y * width + idx_x
    sum_pol_events = np.bincount(
            idx_flat,
            weights=events_array["p"].astype(int) * 2 - 1,
            minlength=width * height,
        )
    sum_events = np.bincount(
            idx_flat,
            minlength=width * height,
        )
    mean = sum_pol_events / (sum_events + 1e-2)

    return np.clip(np.round(mean), -clip, clip)


def get_actor_display_name(actor, truncate=250):
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[: truncate - 1] + "\u2026") if len(name) > truncate else name


class CameraManager(object):
    def __init__(self, parent_actor, width, height, gamma_correction, fov=90.0):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.dim = (width, height)
        self.recording = False
        self.sensor_data = None
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType

        if not self._parent.type_id.startswith("walker.pedestrian"):
            self._camera_transforms = [
                (
                    carla.Transform(
                        carla.Location(
                            x=-2.0 * bound_x, y=+0.0 * bound_y, z=2.0 * bound_z
                        ),
                        carla.Rotation(pitch=8.0),
                    ),
                    Attachment.SpringArmGhost,
                ),
                (
                    carla.Transform(
                        carla.Location(
                            x=+0.8 * bound_x, y=+0.0 * bound_y, z=0.75 * bound_z
                        )
                    ),
                    Attachment.Rigid,
                ),
                (
                    carla.Transform(
                        carla.Location(
                            x=+1.9 * bound_x, y=+1.0 * bound_y, z=1.2 * bound_z
                        )
                    ),
                    Attachment.SpringArmGhost,
                ),
                (
                    carla.Transform(
                        carla.Location(
                            x=-2.8 * bound_x, y=+0.0 * bound_y, z=4.6 * bound_z
                        ),
                        carla.Rotation(pitch=6.0),
                    ),
                    Attachment.SpringArmGhost,
                ),
                (
                    carla.Transform(
                        carla.Location(x=-1.0, y=-1.0 * bound_y, z=0.4 * bound_z)
                    ),
                    Attachment.Rigid,
                ),
            ]
        else:
            self._camera_transforms = [
                (
                    carla.Transform(
                        carla.Location(x=-2.5, z=0.0), carla.Rotation(pitch=-8.0)
                    ),
                    Attachment.SpringArmGhost,
                ),
                (
                    carla.Transform(
                        carla.Location(x=-10.6, y=0.0, z=-1.2),
                        carla.Rotation(pitch=0.5),
                    ),
                    Attachment.Rigid,
                ),
                (
                    carla.Transform(
                        carla.Location(x=2.5, y=0.5, z=1.0), carla.Rotation(pitch=-8.0)
                    ),
                    Attachment.SpringArmGhost,
                ),
                (
                    carla.Transform(
                        carla.Location(x=-4.0, z=2.0), carla.Rotation(pitch=6.0)
                    ),
                    Attachment.SpringArmGhost,
                ),
                (
                    carla.Transform(
                        carla.Location(x=0, y=-2.5, z=-0.0), carla.Rotation(yaw=90.0)
                    ),
                    Attachment.Rigid,
                ),
            ]

        self.transform_index = 1
        self.sensors = [
            ["sensor.camera.rgb", cc.Raw, "Camera RGB", {}],
            ["sensor.camera.depth", cc.Raw, "Camera Depth (Raw)", {}],
            ["sensor.camera.depth", cc.Depth, "Camera Depth (Gray Scale)", {}],
            [
                "sensor.camera.depth",
                cc.LogarithmicDepth,
                "Camera Depth (Logarithmic Gray Scale)",
                {},
            ],
            [
                "sensor.camera.semantic_segmentation",
                cc.Raw,
                "Camera Semantic Segmentation (Raw)",
                {},
            ],
            [
                "sensor.camera.semantic_segmentation",
                cc.CityScapesPalette,
                "Camera Semantic Segmentation (CityScapes Palette)",
                {},
            ],
            [
                "sensor.camera.instance_segmentation",
                cc.CityScapesPalette,
                "Camera Instance Segmentation (CityScapes Palette)",
                {},
            ],
            [
                "sensor.camera.instance_segmentation",
                cc.Raw,
                "Camera Instance Segmentation (Raw)",
                {},
            ],
            ["sensor.lidar.ray_cast", None, "Lidar (Ray-Cast)", {"range": "50"}],
            ["sensor.camera.dvs", cc.Raw, "Dynamic Vision Sensor", {}],
            [
                "sensor.camera.rgb",
                cc.Raw,
                "Camera RGB Distorted",
                {
                    "lens_circle_multiplier": "3.0",
                    "lens_circle_falloff": "3.0",
                    "chromatic_aberration_intensity": "0.5",
                    "chromatic_aberration_offset": "0",
                },
            ],
            ["sensor.camera.optical_flow", cc.Raw, "Optical Flow", {}],
            ["sensor.camera.normals", cc.Raw, "Camera Normals", {}],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith("sensor.camera"):
                bp.set_attribute("image_size_x", str(self.dim[0]))
                bp.set_attribute("image_size_y", str(self.dim[1]))
                if bp.has_attribute("gamma"):
                    bp.set_attribute("gamma", str(gamma_correction))
                if bp.has_attribute("fov"):
                    bp.set_attribute("fov", str(fov))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith("sensor.lidar"):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == "range":
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = (
            True
            if self.index is None
            else (
                force_respawn or (self.sensors[index][2] != self.sensors[self.index][2])
            )
        )
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1],
            )
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda image: CameraManager._parse_image(weak_self, image)
            )

        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith("sensor.lidar"):
            points = np.frombuffer(image.raw_data, dtype=np.dtype("f4"))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.dim[0], 0.5 * self.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.dim[0], self.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith("sensor.camera.dvs"):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(
                image.raw_data,
                dtype=np.dtype(
                    [
                        ("x", np.uint16),
                        ("y", np.uint16),
                        ("t", np.int64),
                        ("pol", np.bool),
                    ]
                ),
            )
            dvs_events_conv = np.empty((len(dvs_events)), dtype=EVENTS_DTYPE)
            dvs_events_conv["t"] = image.timestamp * 1e3  # Convert to microseconds
            dvs_events_conv["x"] = dvs_events["x"]
            dvs_events_conv["y"] = dvs_events["y"]
            dvs_events_conv["p"] = dvs_events["pol"]
            self.sensor_data = dvs_events_conv
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[
                dvs_events[:]["y"], dvs_events[:]["x"], dvs_events[:]["pol"] * 2
            ] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
        elif self.sensors[self.index][0].startswith("sensor.camera.optical_flow"):
            image = image.get_color_coded_flow()
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk("_out/%08d" % image.frame)
