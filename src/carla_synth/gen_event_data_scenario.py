"""
Simulate a scenario
"""

import carla

import pygame

import os

from src.carla_synth.utils import CameraManager, HUD
from src.config import EVENTS_DTYPE

import random

import numpy as np

from .network_settings import params
from .flow_est import FlowEst
from .mpl_multiproc import NBPlot


WIDTH, HEIGHT = 304, 240

FOV = 65.0


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

def on_collision(event, ct, t0):
        print("Collision detected")
        print(event)
        if ct[0] is None:
            ct[0] = int(event.timestamp * 1e3) - t0


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

    t0_dvs = int(world.get_snapshot().timestamp.elapsed_seconds * 1e3)

    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

    blueprint_library = world.get_blueprint_library()

    bp_agent = blueprint_library.filter("vehicle")[0]
    # idx_tf = 48  # random.choice(range(len(world.get_map().get_spawn_points())))
    # transform_agent = world.get_map().get_spawn_points()[idx_tf]

    vehicle_agent = world.spawn_actor(bp_agent, spawn_point_agent)

    actor_list.append(vehicle_agent)

    # add collision sensor
    bp_col = blueprint_library.find("sensor.other.collision")
    col_sensor = world.spawn_actor(bp_col, carla.Transform(), attach_to=vehicle_agent)
    actor_list.append(col_sensor)
    
    # this is so bad...use a list to store the collision time in mutable form
    # so that we can access it from the callback function on_collision.
    collision_time = [None]

    col_sensor.listen(lambda event: on_collision(event, collision_time, t0_dvs))

    hud = HUD(WIDTH, HEIGHT)

    camera_manager = CameraManager(vehicle_agent, hud, 2.2, FOV)
    camera_manager.set_sensor(9, notify=False, force_respawn=True)

    bp_car_npc = blueprint_library.filter("vehicle")[0]

    vehicle_npc = world.spawn_actor(bp_car_npc, spawn_point_npc)

    # we can not provide direct references to the actors created in this function
    # from outside using the directions list, so we create a map to reference them using strings.
    vehicle_map = {"vehicle_agent": vehicle_agent, "vehicle_npc": vehicle_npc}

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

    event_recording = np.empty((0), dtype=EVENTS_DTYPE)

    #### the live sim stuff

    # flowest = FlowEst(params)
    # pl_s = NBPlot(mode="line")

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

        if len(directions) > 0 and t >= directions[0][4]:
            print(f"Executing direction: {directions[0]}")
            _vehicle = vehicle_map[directions[0][0]]
            _func = getattr(_vehicle, directions[0][1])
            _func(*directions[0][2], **directions[0][3])
            directions.pop(0)

        world.tick()

        camera_manager.render(display)

        dvs_events = camera_manager.sensor_data
        if dvs_events is not None:
            event_recording = np.append(event_recording, dvs_events)

            """
            img = np.zeros((HEIGHT, WIDTH), dtype=np.int16)
            img[dvs_events["y"], dvs_events["x"]] = dvs_events["p"] * 2 - 1
            flowest.step(img)

            v_s_l = np.maximum(0.0, flowest.v_s[:, : flowest.v_s.shape[1] // 2].mean())
            v_s_r = np.maximum(0.0, flowest.v_s[:, flowest.v_s.shape[1] // 2 :].mean())

            #pl.plot(flowest.v_s)

            #pl_inp.plot(flowest.spike_p)

            pl_s.plot(
                np.array(
                    [
                        (1.0 - np.exp(-10000.0 * v_s_l))
                        * (1.0 - np.exp(-10000.0 * v_s_r))
                        * 0.5
                        * (v_s_l + v_s_r) * 1000.,
                        0.0,
                    ]
                )
            )"""

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

    print("collision time: ", collision_time[0])

    return event_recording, collision_time[0]
