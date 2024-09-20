import carla
import pygame

from src.carla_synth.camera_manager import CameraManager
from src.carla_synth.recorders import EventRecorder, MetaDataRecorder


from .settings import base_fold_input_data

# from src.carla_synth.scenario_manager import CrossingScenarioManager as ScenarioManager
from src.carla_synth.scenario_manager import (
    NormalDrivingScenarioManager as ScenarioManager,
)

SAVE_EXAMPLES = True

USE_PYGAME = True

N_SAMPLES = 100

WIDTH, HEIGHT = 640, 480

DT = 0.01

POS_CAM = carla.Location(x=2.5, y=0.0, z=1.0)
ROT_CAM = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)

DVS_SENSOR_PARAMS = {
    "refractory_period_ns": 0.001e9,
    "positive_threshold": 0.2,
    "negative_threshold": 0.2,
    "log_eps": 1e-1,
}

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

world = client.get_world()

_settings = world.get_settings()

world.apply_settings(
    carla.WorldSettings(
        no_rendering_mode=False,
        synchronous_mode=True,
        fixed_delta_seconds=DT,
    )
)

mp = world.get_map()

blueprint_library = world.get_blueprint_library()

traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)


scenario_mng = ScenarioManager(
    world, traffic_manager, dt=DT, max_reps=N_SAMPLES, n_vehicles=50, n_pedestrians=0
)

# scenario_mng = ScenarioManager(
#    world, dt=DT, max_reps=N_SAMPLES)


scenario_mng.create_scenario()

# cam_mg = CameraManager(WIDTH, HEIGHT, world, scenario_mng.ego)

cam_dvs_mg = CameraManager(
    WIDTH,
    HEIGHT,
    world,
    scenario_mng.ego,
    sensortype="dvs",
    sensor_params=DVS_SENSOR_PARAMS,
    using_pygame=USE_PYGAME,
)

event_recorder = EventRecorder(DT, base_fold_input_data, WIDTH, HEIGHT)

metadata_recorder = MetaDataRecorder(
    DT, base_fold_input_data, scenario_mng, cam_dvs_mg.camera
)

cam_dvs_mg.set_recorder(event_recorder)
scenario_mng.set_event_recorder(event_recorder)
scenario_mng.set_metadata_recorder(metadata_recorder)


############################
if USE_PYGAME:
    pygame.init()

    display = pygame.display.set_mode(
        (WIDTH, HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF
    )

running = True

while running:
    running = not scenario_mng.finished

    if USE_PYGAME:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    running = False

        display.blit(cam_dvs_mg.surface, (0, 0))

        pygame.display.flip()

    world.tick()

    scenario_mng.update()


if USE_PYGAME:
    pygame.quit()

scenario_mng.destroy_actors()

# cam_mg.destroy()
cam_dvs_mg.destroy()

world.apply_settings(_settings)
