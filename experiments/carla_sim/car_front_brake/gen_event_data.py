"""
Simulate a scenario
"""

import os

import carla

import numpy as np

import pygame

from src.carla_synth.gen_event_data_scenario import gen_event_data_npc_car

from time import time_ns

from itertools import product

base_fold = os.path.join(
    os.path.dirname(__file__), "../../../data/carla_sim/car_front_brake/"
)

WIDTH, HEIGHT = 304, 240

np.random.seed(time_ns() % 2**32)

N_EXAMPLES_PER_VEHICLE_CLASS = 1

T_BRAKE = 10.0
T_END = 15.0
DT = 0.01

TARGET_VEL_KMH = 15.0  # km/h
MIN_DIST_LEADING_VEHICLE = 10.0  # meters

SAVE = True

N_EXTRA_NPC_VEHICLES = 20

VEHICLE_CLASSES = {
    "cars": "car",
    "two_wheel": ["motorcycle", "bicycle"],
    "trucks": [
        "truck",
        "Bus",
        "van",
    ],  # for some reason the blueprint library has only "Bus" capitalized
}


def run_car_front_no_brake(
    t_end,
    dt,
    target_vel_kmh,
    min_dist_leading_vehicle,
    n_extra_npc_vehicles,
    vehicle_class,
    vehicle_npc_idx=None,
    index_example=0,
    save=True,
    pg_display=None,
    dim=(WIDTH, HEIGHT),
):
    directions = [
        ("set_autopilot", ["vehicle_npc", True], {}, 0.0),
        ("set_autopilot", ["vehicle_player", True], {}, 0.0),
        ("set_desired_speed", ["vehicle_npc", target_vel_kmh], {}, 0.0),
        ("set_desired_speed", ["vehicle_player", target_vel_kmh], {}, 0.0),
        (
            "distance_to_leading_vehicle",
            ["vehicle_player", min_dist_leading_vehicle],
            {},
            0.0,
        ),
    ]

    client = carla.Client("localhost", 2000)
    client.set_timeout(2.0)

    world = client.get_world()

    spawn_point_player = world.get_map().get_spawn_points()[48]
    spawn_point_npc = carla.Transform(
        spawn_point_player.location + carla.Location(x=-15.0),
        spawn_point_player.rotation,
    )

    events, ct, vehicle_npc_idx = gen_event_data_npc_car(
        directions,
        t_end,
        dt,
        spawn_point_player,
        spawn_point_npc,
        n_extra_npc_vehicles,
        vehicle_npc_idx=vehicle_npc_idx,
        vehicle_npc_class=VEHICLE_CLASSES[vehicle_class],
        pg_display=pg_display,
        dim=dim,
    )

    if save:
        save_fold = os.path.join(base_fold, vehicle_class, f"example_{index_example}")
        if not os.path.exists(save_fold):
            os.makedirs(save_fold)

        np.save(os.path.join(save_fold, "events_baseline.npy"), events)
        np.savez(
            os.path.join(save_fold, "sim_data_baseline.npz"),
            collision_time=ct,
            t_end=t_end,
            dt=dt,
            target_vel_kmh=TARGET_VEL_KMH,
        )
    
    return vehicle_npc_idx


def run_car_front_brake(
    t_end,
    t_brake,
    dt,
    target_vel_kmh,
    min_dist_leading_vehicle,
    n_extra_npc_vehicles,
    vehicle_class,
    vehicle_npc_idx=None,
    index_example=0,
    save=True,
    pg_display=None,
    dim=(WIDTH, HEIGHT),
):
    directions = [
        ("set_autopilot", ["vehicle_npc", True], {}, 0.0),
        ("set_autopilot", ["vehicle_player", True], {}, 0.0),
        ("set_desired_speed", ["vehicle_npc", target_vel_kmh], {}, 0.0),
        ("set_desired_speed", ["vehicle_player", target_vel_kmh], {}, 0.0),
        (
            "distance_to_leading_vehicle",
            ["vehicle_player", min_dist_leading_vehicle],
            {},
            0.0,
        ),
        ("set_autopilot", ["vehicle_npc", False], {}, t_brake),
        ("ignore_vehicles_percentage", ["vehicle_player", 100], {}, t_brake),
        (
            "apply_control",
            ["vehicle_npc", carla.VehicleControl(throttle=0.0, brake=2.0)],
            {},
            t_brake,
        ),
    ]

    client = carla.Client("localhost", 2000)
    client.set_timeout(2.0)

    world = client.get_world()

    spawn_point_player = world.get_map().get_spawn_points()[48]
    spawn_point_npc = carla.Transform(
        spawn_point_player.location + carla.Location(x=-15.0),
        spawn_point_player.rotation,
    )

    events, ct, vehicle_npc_idx = gen_event_data_npc_car(
        directions,
        t_end,
        dt,
        spawn_point_player,
        spawn_point_npc,
        n_extra_npc_vehicles,
        vehicle_npc_idx=vehicle_npc_idx,
        vehicle_npc_class=VEHICLE_CLASSES[vehicle_class],
        pg_display=pg_display,
        dim=dim,
    )

    if save:
        save_fold = os.path.join(base_fold, vehicle_class, f"example_{index_example}")
        if not os.path.exists(save_fold):
            os.makedirs(save_fold)

        np.save(os.path.join(save_fold, "events.npy"), events)
        np.savez(
            os.path.join(save_fold, "sim_data.npz"),
            collision_time=ct,
            t_end=t_end,
            dt=dt,
            target_vel_kmh=TARGET_VEL_KMH,
            t_brake=t_brake,
        )
    
    return vehicle_npc_idx


# pygame.init()

# display = pygame.display.set_mode((WIDTH, HEIGHT))

for vehicle_class, i in product(VEHICLE_CLASSES.keys(), range(N_EXAMPLES_PER_VEHICLE_CLASS)):
    print(f"Running example {i} for vehicle class {vehicle_class}")
    _vehicle_npc_idx = run_car_front_brake(
        T_END,
        T_BRAKE,
        DT,
        TARGET_VEL_KMH,
        MIN_DIST_LEADING_VEHICLE,
        N_EXTRA_NPC_VEHICLES,
        vehicle_class,
        index_example=i,
        save=SAVE,
        # pg_display=display,
    )

    run_car_front_no_brake(
        T_END,
        DT,
        TARGET_VEL_KMH,
        MIN_DIST_LEADING_VEHICLE,
        N_EXTRA_NPC_VEHICLES,
        vehicle_class,
        index_example=i,
        vehicle_npc_idx=_vehicle_npc_idx,
        save=SAVE,
        # pg_display=display,
    )

# pygame.quit()

