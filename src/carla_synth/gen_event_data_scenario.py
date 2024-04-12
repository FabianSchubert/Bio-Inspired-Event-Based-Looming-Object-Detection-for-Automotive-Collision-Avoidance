"""
Simulate a scenario
"""

import carla
from agents.navigation.basic_agent import BasicAgent

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
    spawn_point_player: carla.Transform,
    spawn_point_npc: carla.Transform,
    n_bg_vehicles: int,
    vehicle_player_idx: int = 0,
    vehicle_npc_idx: int = 0,
    t_cutoff: int = 0,
):
    t_cutoff = int(t_cutoff * 1e3)

    pygame.font.init()

    client = carla.Client("localhost", 2000)
    client.set_timeout(2.0)

    actor_list = []

    world = client.get_world()

    world_settings = world.get_settings()
    world_settings.synchronous_mode = True  # Enables synchronous mode
    world_settings.fixed_delta_seconds = dt_secs
    world.apply_settings(world_settings)

    t0_dvs = int(world.get_snapshot().timestamp.elapsed_seconds * 1e3) + t_cutoff

    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

    blueprint_library = world.get_blueprint_library()

    bp_player = blueprint_library.filter("vehicle")[vehicle_player_idx]
    # idx_tf = 48  # random.choice(range(len(world.get_map().get_spawn_points())))
    # transform_agent = world.get_map().get_spawn_points()[idx_tf]

    vehicle_player = world.spawn_actor(bp_player, spawn_point_player)

    actor_list.append(vehicle_player)

    agent_player = BasicAgent(vehicle_player)

    # add collision sensor
    bp_col = blueprint_library.find("sensor.other.collision")
    col_sensor = world.spawn_actor(bp_col, carla.Transform(), attach_to=vehicle_player)
    actor_list.append(col_sensor)

    # this is so bad...use a list to store the collision time in mutable form
    # so that we can access it from the callback function on_collision.
    collision_time = [None]

    col_sensor.listen(lambda event: on_collision(event, collision_time, t0_dvs))

    hud = HUD(WIDTH, HEIGHT)

    camera_manager = CameraManager(vehicle_player, hud, 2.2, FOV)
    camera_manager.set_sensor(9, notify=False, force_respawn=True)

    bp_car_npc = blueprint_library.filter("vehicle")[vehicle_npc_idx]

    vehicle_npc = world.spawn_actor(bp_car_npc, spawn_point_npc)

    actor_list.append(vehicle_npc)

    print(f"Vehicle NPC: {vehicle_npc}, index: {vehicle_npc_idx}")

    agent_npc = BasicAgent(vehicle_npc)

    # we can not provide direct references to the actors and agents created in this function
    # from outside using the directions list, so we create a map to reference them using strings.
    obj_map = {
        "vehicle_player": vehicle_player,
        "vehicle_npc": vehicle_npc,
        "agent_player": agent_player,
        "agent_npc": agent_npc,
    }

    directions_map = {
        "set_autopilot": lambda args, kwargs: obj_map[args[0]].set_autopilot(args[1]),
        "apply_control": lambda args, kwargs: obj_map[args[0]].apply_control(args[1]),
        "set_desired_speed": lambda args, kwargs: tm.set_desired_speed(
            obj_map[args[0]], args[1] # args[1] is the desired speed in km/h
        ),
        "ignore_vehicles_percentage": lambda args, kwargs: tm.ignore_vehicles_percentage(
            obj_map[args[0]], args[1] # args[1] is the percentage
        ),
        "distance_to_leading_vehicle": lambda args, kwargs: tm.distance_to_leading_vehicle(
            obj_map[args[0]], args[1] # args[1] is the distance in meters
        ),
    }

    tm.auto_lane_change(vehicle_player, False)  # disable auto lane change
    tm.ignore_lights_percentage(vehicle_player, 100)  # ignore traffic lights
    tm.ignore_signs_percentage(vehicle_player, 100)  # ignore traffic signs
    tm.ignore_walkers_percentage(vehicle_player, 100)  # ignore pedestrians

    tm.auto_lane_change(vehicle_npc, False)  # disable auto lane change
    tm.ignore_lights_percentage(vehicle_npc, 100)  # ignore traffic lights
    tm.ignore_signs_percentage(vehicle_npc, 100)  # ignore traffic signs
    tm.ignore_walkers_percentage(vehicle_npc, 100)  # ignore pedestrians
    #tm.ignore_vehicles_percentage(vehicle_player, 100)  # ignore other vehicles

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

    idx_direction = 0

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

        if idx_direction < len(directions):
            if t >= directions[idx_direction][3]:
                print(f"Executing direction: {directions[idx_direction]}")
                _func = directions_map[directions[idx_direction][0]]
                _func(directions[idx_direction][1], directions[idx_direction][2])
                idx_direction += 1

        world.tick()

        # vehicle_player.apply_control(agent_player.run_step())
        # vehicle_npc.apply_control(agent_npc.run_step())

        camera_manager.render(display)

        dvs_events = camera_manager.sensor_data
        if dvs_events is not None:
            event_recording = np.append(event_recording, dvs_events)

        pygame.display.flip()

    event_recording["t"] -= t0_dvs

    event_recording = event_recording[event_recording["t"] > 0]

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
