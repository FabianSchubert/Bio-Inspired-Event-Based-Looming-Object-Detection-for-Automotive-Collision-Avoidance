
import carla
import numpy as np
import pygame

from src.carla_synth.utils import convert_events, downsample_events, pol_evt_img_to_rgb

class CameraManager:
    def __init__(self, width, height, world, car, sensortype="rgb", sensor_params={}, using_pygame=True):
        self.width = width
        self.height = height

        self.senor_params = sensor_params

        self.world = world

        self.bpl = self.world.get_blueprint_library()

        self.using_pygame = using_pygame

        self.sensortype = sensortype

        assert sensortype in ["rgb", "dvs"]

        if sensortype == "rgb":
            cam_bp = self.bpl.find("sensor.camera.rgb")

        else:
            cam_bp = self.bpl.find("sensor.camera.dvs")

        self.event_data = None

        cam_bp.set_attribute("image_size_x", str(width))
        cam_bp.set_attribute("image_size_y", str(height))

        for key, value in sensor_params.items():
            cam_bp.set_attribute(key, str(value))

        pos_cam = carla.Location(x=2.5, y=0.0, z=1.0)
        rot_cam = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)

        self.camera = self.world.spawn_actor(
            cam_bp, carla.Transform(pos_cam, rot_cam), attach_to=car
        )

        self.rgb_arr = np.zeros((height, width, 3), dtype=np.uint8)

        if self.using_pygame:
            self.surface = pygame.surfarray.make_surface(self.rgb_arr.swapaxes(0, 1))
        else:
            self.surface = None

        if sensortype == "rgb":
            self.camera.listen(self.udpate_callback_rgb)
        else:
            self.camera.listen(self.udpate_callback_dvs)

        self.recorder = None

    def udpate_callback_rgb(self, data):
        if self.using_pygame:
            img = np.reshape(np.copy(data.raw_data), (self.height, self.width, 4))
            img = img[:, :, :3]
            img = img[:, :, ::-1]
            self.rgb_arr[:] = img
            self.surface = pygame.surfarray.make_surface(self.rgb_arr.swapaxes(0, 1))

        if self.recorder is not None:
            self.recorder.receive_data(data)

    def udpate_callback_dvs(self, data):
        evt_arr = convert_events(data)
        events_binned = downsample_events(evt_arr, self.width, self.height)
        if self.using_pygame:
            self.rgb_arr[:] = pol_evt_img_to_rgb(
                events_binned.reshape((self.height, self.width))
            )

            self.surface = pygame.surfarray.make_surface(self.rgb_arr.swapaxes(0, 1))

        if self.recorder is not None:
            self.recorder.receive_data(events_binned)

    def destroy(self):
        self.camera.destroy()

    def set_recorder(self, recorder):
        self.recorder = recorder

