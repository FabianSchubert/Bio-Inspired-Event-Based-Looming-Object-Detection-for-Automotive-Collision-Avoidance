"""
Simulate a scenario
"""

import os

import carla

import numpy as np

from src.carla_synth.gen_event_data_scenario import gen_event_data_npc_car

base_fold = os.path.join(os.path.dirname(__file__), "../../../data/carla_sim/car_front_brake/")

T_BRAKE = 5.0
T_END = 7.0
DT = 0.001

# Define the scenario
directions = [
    ("vehicle_npc", "set_autopilot", [True], {}, 0.0),
    ("vehicle_agent", "set_autopilot", [True], {}, 0.0),
    ("vehicle_npc", "set_autopilot", [False], {}, T_BRAKE),
    ("vehicle_agent", "set_autopilot", [False], {}, T_BRAKE),
    (
        "vehicle_npc", "apply_control",
        [carla.VehicleControl(throttle=0.0, brake=2.0)],
        {},
        T_BRAKE,
    ),
]

client = carla.Client("localhost", 2000)
client.set_timeout(2.0)

world = client.get_world()

spawn_point_agent = world.get_map().get_spawn_points()[48]
spawn_point_npc = carla.Transform(spawn_point_agent.location + carla.Location(x=-10.0), spawn_point_agent.rotation)

events, ct = gen_event_data_npc_car(directions, T_END, DT, spawn_point_agent, spawn_point_npc, 10)

np.save(os.path.join(base_fold, "events.npy"), events)
np.savez(os.path.join(base_fold, "sim_data.npz"), collision_time=ct, t_end=T_END, dt=DT)


'''
import carla

import pygame

import os

from src.carla_synth.utils import CameraManager, HUD
from src.config import EVENTS_DTYPE

import random

import numpy as np

WIDTH, HEIGHT = 304, 240


# a list of tuples, each containing a command to execute, the arguments to
# pass to that command, the keyword arguments to pass to that command, and
# the time at which to execute that command.
# Each time the simulation time exceeds the time given for the
# first command in the list, it is executed and removed from the list.
# This means that the commands should be in order of increasing time.
"""
directions = [
    (vehicle_npc.set_autopilot, [True], {}, 1.0),
    (vehicle_agent.set_autopilot, [True], {}, 1.0),
    (vehicle_npc.set_autopilot, [False], {}, 7.0),
    (vehicle_agent.set_autopilot, [False], {}, 7.0),
    (
        vehicle_npc.apply_control,
        [carla.VehicleControl(throttle=0.0, brake=2.0)],
        {},
        4.0,
    ),
]"""


def gen_event_data_npc_car(
    directions: list,
    t_run: float,
    dt_secs: float,
    spawn_point_agent: carla.Transform,
    spawn_point_npc: carla.Transform,
    n_bg_vehicles: int,
):
    pygame.font.init()

    client = carla.Client("localhost", 2000)
    client.set_timeout(2.0)

    actor_list = []

    world = client.get_world()

    world_settings = world.get_settings()
    world_settings.synchronous_mode = True  # Enables synchronous mode
    world_settings.fixed_delta_seconds = dt_secs
    world.apply_settings(world_settings)

    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

    blueprint_library = world.get_blueprint_library()

    bp_agent = blueprint_library.filter("vehicle")[0]
    # idx_tf = 48  # random.choice(range(len(world.get_map().get_spawn_points())))
    # transform_agent = world.get_map().get_spawn_points()[idx_tf]

    vehicle_agent = world.spawn_actor(bp_agent, spawn_point_agent)

    actor_list.append(vehicle_agent)

    hud = HUD(WIDTH, HEIGHT)

    camera_manager = CameraManager(vehicle_agent, hud, 2.2)
    camera_manager.set_sensor(9, notify=False, force_respawn=True)

    bp_car_npc = blueprint_library.filter("vehicle")[0]

    vehicle_npc = world.spawn_actor(bp_car_npc, spawn_point_npc)

    actor_list.append(vehicle_npc)

    spawn_points = world.get_map().get_spawn_points()
    n_spawn_points = len(spawn_points)

    if n_bg_vehicles < n_spawn_points:
        random.shuffle(spawn_points)
        spawn_points = spawn_points[:n_bg_vehicles]
    else:
        print("Not enough spawn points for background vehicles")

    # Create some background vehicles

    for i in range(n_bg_vehicles):
        bp_car_bg = random.choice(blueprint_library.filter("vehicle"))
        try:
            vehicle_bg = world.spawn_actor(bp_car_bg, spawn_points[i])
            vehicle_bg.set_autopilot(True)
            actor_list.append(vehicle_bg)
        except Exception as e:
            print(f"Failed to spawn vehicle {i}")

    #### main loop

    pygame.init()

    display = pygame.display.set_mode(
        (WIDTH, HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF
    )

    display.fill((0, 0, 0))

    t = 0.0

    t0_dvs = world.get_snapshot().timestamp.elapsed_seconds * 1e3

    event_recording = np.empty((0), dtype=EVENTS_DTYPE)

    running = True

    while t < t_run:
        t += dt_secs

        # print(f"t: {t}")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

                print("Exiting...")

                running = False

        if not running:
            break

        if len(directions) > 0 and t >= directions[0][3]:
            print(f"Executing direction: {directions[0]}")
            directions[0][0](*directions[0][1], **directions[0][2])
            directions.pop(0)

        world.tick()

        camera_manager.render(display)

        dvs_events = camera_manager.sensor_data
        if dvs_events is not None:
            event_recording = np.append(event_recording, dvs_events)

        pygame.display.flip()

    event_recording["t"] -= t0_dvs

    for actor in actor_list:
        actor.destroy()

    camera_manager.sensor.destroy()
    camera_manager.sensor = None
    camera_manager.index = None

    world.tick()

    tm.set_synchronous_mode(False)

    # world_settings = world.get_settings()
    world_settings.synchronous_mode = False
    world_settings.fixed_delta_seconds = None
    world.apply_settings(world_settings)

    return event_recording
'''