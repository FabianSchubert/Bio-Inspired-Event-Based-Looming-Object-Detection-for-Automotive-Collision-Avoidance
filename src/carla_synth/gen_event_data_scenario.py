"""
Simulate a scenario
"""

from typing import Union

import carla

import pygame

from agents.navigation.basic_agent import BasicAgent

from src.carla_synth.utils import CameraManager
from src.config import EVENTS_DTYPE

import random

import numpy as np


WIDTH, HEIGHT = 304, 240

FOV = 65.0


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
    vehicle_npc_idx: Union[int, None] = None,
    vehicle_npc_class: Union[str, list] = "car",
    t_cutoff: int = 0,
    pg_display=None,
    dim=(WIDTH, HEIGHT),
):
    t_cutoff = int(t_cutoff * 1e3)

    client = carla.Client("localhost", 2000)
    client.set_timeout(15.0)

    print(f"Carla Client Version: {client.get_client_version()}")
    print(f"Carla Server Version: {client.get_server_version()}")

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

    agent_player = BasicAgent(vehicle_player)

    actor_list.append(vehicle_player)

    # add collision sensor
    bp_col = blueprint_library.find("sensor.other.collision")
    col_sensor = world.spawn_actor(bp_col, carla.Transform(), attach_to=vehicle_player)
    actor_list.append(col_sensor)

    # this is so bad...use a list to store the collision time in mutable form
    # so that we can access it from the callback function on_collision.
    collision_time = [None]

    col_sensor.listen(lambda event: on_collision(event, collision_time, t0_dvs))

    camera_manager = CameraManager(vehicle_player, dim[0], dim[1], 2.2, FOV)
    camera_manager.set_sensor(9, notify=False, force_respawn=True)

    if isinstance(vehicle_npc_class, str):
        vehicle_npc_class = [vehicle_npc_class]

    print(f"Vehicle NPC class: {vehicle_npc_class}")

    list_npc_vehicles = [
        bp
        for bp in blueprint_library.filter("vehicle")
        if bp.get_attribute("base_type").as_str() in vehicle_npc_class
    ]
    print(list_npc_vehicles)

    if vehicle_npc_idx is None:
        vehicle_npc_idx = random.choice(range(len(list_npc_vehicles)))
    else:
        vehicle_npc_idx = vehicle_npc_idx % len(
            list_npc_vehicles
        )  # make sure the index is within the range by taking the modulo (cyclic)

    bp_car_npc = list_npc_vehicles[vehicle_npc_idx]

    vehicle_npc = world.spawn_actor(bp_car_npc, spawn_point_npc)

    actor_list.append(vehicle_npc)

    print(f"Vehicle NPC: {vehicle_npc}, index: {vehicle_npc_idx}")

    # we can not provide direct references to the actors and agents created in this function
    # from outside using the directions list, so we create a map to reference them using strings.
    obj_map = {
        "vehicle_player": vehicle_player,
        "vehicle_npc": vehicle_npc,
    }

    directions_map = {
        "set_autopilot": lambda args, kwargs: obj_map[args[0]].set_autopilot(args[1]),
        "apply_control": lambda args, kwargs: obj_map[args[0]].apply_control(args[1]),
        "set_desired_speed": lambda args, kwargs: tm.set_desired_speed(
            obj_map[args[0]],
            args[1],  # args[1] is the desired speed in km/h
        ),
        "ignore_vehicles_percentage": lambda args,
        kwargs: tm.ignore_vehicles_percentage(
            obj_map[args[0]],
            args[1],  # args[1] is the percentage
        ),
        "distance_to_leading_vehicle": lambda args,
        kwargs: tm.distance_to_leading_vehicle(
            obj_map[args[0]],
            args[1],  # args[1] is the distance in meters
        ),
        "set_path": lambda args, kwargs: tm.set_path(obj_map[args[0]], args[1]),
    }

    tm.auto_lane_change(vehicle_player, False)  # disable auto lane change
    tm.ignore_lights_percentage(vehicle_player, 100)  # ignore traffic lights
    tm.ignore_signs_percentage(vehicle_player, 100)  # ignore traffic signs
    tm.ignore_walkers_percentage(vehicle_player, 100)  # ignore pedestrians

    tm.auto_lane_change(vehicle_npc, False)  # disable auto lane change
    tm.ignore_lights_percentage(vehicle_npc, 100)  # ignore traffic lights
    tm.ignore_signs_percentage(vehicle_npc, 100)  # ignore traffic signs
    tm.ignore_walkers_percentage(vehicle_npc, 100)  # ignore pedestrians
    tm.ignore_vehicles_percentage(vehicle_npc, 100)  # ignore other vehicles

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

    #world.get_spectator().set_transform(vehicle_player.get_transform())

    #### main loop
    show_sim = False
    if pg_display is not None:
        show_sim = True
        pygame.init()
        display = pg_display

        display.fill((0, 0, 0))

    t = 0.0

    event_recording = np.empty((0), dtype=EVENTS_DTYPE)

    #### the live sim stuff

    running = True

    idx_direction = 0

    while t < t_run and collision_time[0] is None:
        t += dt_secs

        # print(f"t: {t}")

        if show_sim:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
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

        if show_sim:
            camera_manager.render(display)

        dvs_events = camera_manager.sensor_data
        if dvs_events is not None:
            event_recording = np.append(event_recording, dvs_events)

        if show_sim:
            pygame.display.flip()

    event_recording["t"] -= t0_dvs

    event_recording = event_recording[event_recording["t"] > 0]

    for actor in actor_list:
        actor.destroy()

    camera_manager.sensor.destroy()
    camera_manager.sensor = None
    camera_manager.index = None

    # tm.set_synchronous_mode(False)

    # world_settings = world.get_settings()
    # world_settings.synchronous_mode = False
    # world_settings.fixed_delta_seconds = None
    # world.apply_settings(world_settings)

    print("collision time: ", collision_time[0])

    return event_recording, collision_time[0], vehicle_npc_idx


# pedestrian scenario
def gen_event_data_npc_pedestrian(
    directions: list,
    t_run: float,
    dt_secs: float,
    spawn_point_player: carla.Transform,
    spawn_point_npc: carla.Transform,
    n_bg_vehicles: int,
    vehicle_player_idx: int = 0,
    npc_idx: Union[int, None] = None,
    t_cutoff: int = 0,
    pg_display=None,
    dim=(WIDTH, HEIGHT),
):
    t_cutoff = int(t_cutoff * 1e3)

    client = carla.Client("localhost", 2000)
    client.set_timeout(15.0)

    print(f"Carla Client Version: {client.get_client_version()}")
    print(f"Carla Server Version: {client.get_server_version()}")

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

    # add collision sensor
    bp_col = blueprint_library.find("sensor.other.collision")
    col_sensor = world.spawn_actor(bp_col, carla.Transform(), attach_to=vehicle_player)
    actor_list.append(col_sensor)

    # this is so bad...use a list to store the collision time in mutable form
    # so that we can access it from the callback function on_collision.
    collision_time = [None]

    col_sensor.listen(lambda event: on_collision(event, collision_time, t0_dvs))

    camera_manager = CameraManager(vehicle_player, dim[0], dim[1], 2.2, FOV)
    camera_manager.set_sensor(9, notify=False, force_respawn=True)

    list_npc_pedestrians = blueprint_library.filter("walker.pedestrian.*")
    # print(list_npc_pedestrians)

    if npc_idx is None:
        npc_idx = random.choice(range(len(list_npc_pedestrians)))
    else:
        npc_idx = npc_idx % len(
            list_npc_pedestrians
        )  # make sure the index is within the range by taking the modulo (cyclic)

    bp_npc = list_npc_pedestrians[npc_idx]

    # set is_invincible to false, for some reason
    # collisions are not detected when it is set to true
    if bp_npc.has_attribute("is_invincible"):
        print("set is_invincible to false")
        bp_npc.set_attribute("is_invincible", "false")

    pedestrian_npc = world.spawn_actor(bp_npc, spawn_point_npc)

    actor_list.append(pedestrian_npc)

    print(f"Pedestrion NPC: {pedestrian_npc}, index: {npc_idx}")

    # we can not provide direct references to the actors and agents created in this function
    # from outside using the directions list, so we create a map to reference them using strings.
    obj_map = {
        "vehicle_player": vehicle_player,
        "pedestrian_npc": pedestrian_npc,
    }

    directions_map = {
        "set_autopilot": lambda args, kwargs: obj_map[args[0]].set_autopilot(args[1]),
        "apply_control": lambda args, kwargs: obj_map[args[0]].apply_control(args[1]),
        "set_desired_speed": lambda args, kwargs: tm.set_desired_speed(
            obj_map[args[0]],
            args[1],  # args[1] is the desired speed in km/h
        ),
        "ignore_vehicles_percentage": lambda args,
        kwargs: tm.ignore_vehicles_percentage(
            obj_map[args[0]],
            args[1],  # args[1] is the percentage
        ),
        "ignore_walkers_percentage": lambda args, kwargs: tm.ignore_walkers_percentage(
            obj_map[args[0]],
            args[1],  # args[1] is the percentage
        ),
        "distance_to_leading_vehicle": lambda args,
        kwargs: tm.distance_to_leading_vehicle(
            obj_map[args[0]],
            args[1],  # args[1] is the distance in meters
        ),
    }

    tm.auto_lane_change(vehicle_player, False)  # disable auto lane change
    tm.ignore_lights_percentage(vehicle_player, 100)  # ignore traffic lights
    tm.ignore_signs_percentage(vehicle_player, 100)  # ignore traffic signs
    tm.ignore_walkers_percentage(vehicle_player, 100)  # ignore pedestrians

    # tm.auto_lane_change(pedestrian_npc, False)  # disable auto lane change
    # tm.ignore_lights_percentage(vehicle_npc, 100)  # ignore traffic lights
    # tm.ignore_signs_percentage(vehicle_npc, 100)  # ignore traffic signs
    # tm.ignore_walkers_percentage(vehicle_npc, 100)  # ignore pedestrians
    # tm.ignore_vehicles_percentage(vehicle_player, 100)  # ignore other vehicles

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

    world.get_spectator().set_transform(vehicle_player.get_transform())

    #### main loop
    show_sim = False
    if pg_display is not None:
        show_sim = True
        pygame.init()
        display = pg_display

        display.fill((0, 0, 0))

    t = 0.0

    event_recording = np.empty((0), dtype=EVENTS_DTYPE)

    #### the live sim stuff

    running = True

    idx_direction = 0

    while t < t_run and collision_time[0] is None:
        t += dt_secs

        # print(f"t: {t}")

        if show_sim:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
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

        if show_sim:
            camera_manager.render(display)

        dvs_events = camera_manager.sensor_data
        if dvs_events is not None:
            event_recording = np.append(event_recording, dvs_events)

        if show_sim:
            pygame.display.flip()

    event_recording["t"] -= t0_dvs

    event_recording = event_recording[event_recording["t"] > 0]

    for actor in actor_list:
        actor.destroy()

    camera_manager.sensor.destroy()
    camera_manager.sensor = None
    camera_manager.index = None

    # tm.set_synchronous_mode(False)

    # world_settings = world.get_settings()
    # world_settings.synchronous_mode = False
    # world_settings.fixed_delta_seconds = None
    # world.apply_settings(world_settings)

    print("collision time: ", collision_time[0])

    return event_recording, collision_time[0], npc_idx


# pedestrian scenario
def gen_event_data_no_npc(
    directions: list,
    t_run: float,
    dt_secs: float,
    spawn_point_player: carla.Transform,
    n_bg_vehicles: int,
    vehicle_player_idx: int = 0,
    t_cutoff: int = 0,
    pg_display=None,
    dim=(WIDTH, HEIGHT),
):
    t_cutoff = int(t_cutoff * 1e3)

    client = carla.Client("localhost", 2000)
    client.set_timeout(15.0)

    print(f"Carla Client Version: {client.get_client_version()}")
    print(f"Carla Server Version: {client.get_server_version()}")

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

    # add collision sensor
    bp_col = blueprint_library.find("sensor.other.collision")
    col_sensor = world.spawn_actor(bp_col, carla.Transform(), attach_to=vehicle_player)
    actor_list.append(col_sensor)

    # this is so bad...use a list to store the collision time in mutable form
    # so that we can access it from the callback function on_collision.
    collision_time = [None]

    col_sensor.listen(lambda event: on_collision(event, collision_time, t0_dvs))

    camera_manager = CameraManager(vehicle_player, dim[0], dim[1], 2.2, FOV)
    camera_manager.set_sensor(9, notify=False, force_respawn=True)

    # we can not provide direct references to the actors and agents created in this function
    # from outside using the directions list, so we create a map to reference them using strings.
    obj_map = {
        "vehicle_player": vehicle_player,
    }

    directions_map = {
        "set_autopilot": lambda args, kwargs: obj_map[args[0]].set_autopilot(args[1]),
        "apply_control": lambda args, kwargs: obj_map[args[0]].apply_control(args[1]),
        "set_desired_speed": lambda args, kwargs: tm.set_desired_speed(
            obj_map[args[0]],
            args[1],  # args[1] is the desired speed in km/h
        ),
        "ignore_vehicles_percentage": lambda args,
        kwargs: tm.ignore_vehicles_percentage(
            obj_map[args[0]],
            args[1],  # args[1] is the percentage
        ),
        "ignore_walkers_percentage": lambda args, kwargs: tm.ignore_walkers_percentage(
            obj_map[args[0]],
            args[1],  # args[1] is the percentage
        ),
        "distance_to_leading_vehicle": lambda args,
        kwargs: tm.distance_to_leading_vehicle(
            obj_map[args[0]],
            args[1],  # args[1] is the distance in meters
        ),
    }

    tm.auto_lane_change(vehicle_player, False)  # disable auto lane change
    tm.ignore_lights_percentage(vehicle_player, 100)  # ignore traffic lights
    tm.ignore_signs_percentage(vehicle_player, 100)  # ignore traffic signs
    tm.ignore_walkers_percentage(vehicle_player, 100)  # ignore pedestrians

    # tm.auto_lane_change(pedestrian_npc, False)  # disable auto lane change
    # tm.ignore_lights_percentage(vehicle_npc, 100)  # ignore traffic lights
    # tm.ignore_signs_percentage(vehicle_npc, 100)  # ignore traffic signs
    # tm.ignore_walkers_percentage(vehicle_npc, 100)  # ignore pedestrians
    # tm.ignore_vehicles_percentage(vehicle_player, 100)  # ignore other vehicles

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

    world.get_spectator().set_transform(vehicle_player.get_transform())

    #### main loop
    show_sim = False
    if pg_display is not None:
        show_sim = True
        pygame.init()
        display = pg_display

        display.fill((0, 0, 0))

    t = 0.0

    event_recording = np.empty((0), dtype=EVENTS_DTYPE)

    #### the live sim stuff

    running = True

    idx_direction = 0

    while t < t_run and collision_time[0] is None:
        t += dt_secs

        # print(f"t: {t}")

        if show_sim:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
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

        if show_sim:
            camera_manager.render(display)

        dvs_events = camera_manager.sensor_data
        if dvs_events is not None:
            event_recording = np.append(event_recording, dvs_events)

        if show_sim:
            pygame.display.flip()

    event_recording["t"] -= t0_dvs

    event_recording = event_recording[event_recording["t"] > 0]

    for actor in actor_list:
        actor.destroy()

    camera_manager.sensor.destroy()
    camera_manager.sensor = None
    camera_manager.index = None

    # tm.set_synchronous_mode(False)

    # world_settings = world.get_settings()
    # world_settings.synchronous_mode = False
    # world_settings.fixed_delta_seconds = None
    # world.apply_settings(world_settings)

    print("collision time: ", collision_time[0])

    return event_recording, collision_time[0]
