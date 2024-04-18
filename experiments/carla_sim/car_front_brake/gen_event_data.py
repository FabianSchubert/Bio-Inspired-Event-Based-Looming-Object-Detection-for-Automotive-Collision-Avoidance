"""
Simulate a scenario
"""

import os

import carla

import numpy as np

import pygame

from src.carla_synth.gen_event_data_scenario import (
    gen_event_data_npc_car,
    gen_event_data_npc_pedestrian,
    gen_event_data_no_npc,
)

from time import time_ns

from itertools import product

base_fold = os.path.join(
    os.path.dirname(__file__), "../../../data/carla_sim/car_front_brake/"
)

WIDTH, HEIGHT = 304, 240

np.random.seed(time_ns() % 2**32)

N_EXAMPLES_PER_VEHICLE_CLASS = 10

T_BRAKE = 10.0 # when the NPC vehicle starts braking
T_CROSS = 3.5 # when the pedestrian starts crossing
T_END = 15.0
DT = 0.01 # 100 Hz

TARGET_VEL_KMH = 15.0  # km/h
MIN_DIST_LEADING_VEHICLE = 10.0  # meters

SAVE = True

N_EXTRA_NPC_VEHICLES_CAR_FRONT = 20
N_EXTRA_NPC_VEHICLES_PEDESTRIAN_CROSS = 0

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
    filename_extension="_baseline",
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

        np.save(os.path.join(save_fold, "events" + filename_extension + ".npy"), events)
        np.savez(
            os.path.join(save_fold, "sim_data" + filename_extension + ".npz"),
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
    filename_extension="",
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

        np.save(os.path.join(save_fold, "events" + filename_extension + ".npy"), events)
        np.savez(
            os.path.join(save_fold, "sim_data" + filename_extension + ".npz"),
            collision_time=ct,
            t_end=t_end,
            dt=dt,
            target_vel_kmh=TARGET_VEL_KMH,
            t_brake=t_brake,
        )

    return vehicle_npc_idx


def run_pedestrian_crossing(
    t_end,
    t_cross,
    dt,
    target_vel_kmh,
    n_extra_npc_vehicles,
    npc_idx=None,
    index_example=0,
    save=True,
    pg_display=None,
    dim=(WIDTH, HEIGHT),
    filename_extension="",
):
    directions = [
        ("set_autopilot", ["vehicle_player", True], {}, 0.0),
        ("set_desired_speed", ["vehicle_player", target_vel_kmh], {}, 0.0),
        ("ignore_walkers_percentage", ["vehicle_player", 100], {}, t_cross),
        (
            "apply_control",
            [
                "pedestrian_npc",
                carla.WalkerControl(direction=carla.Vector3D(0.0, 1.0, 0.0), speed=1.0),
            ],
            {},
            t_cross,
        ),
    ]

    client = carla.Client("localhost", 2000)
    client.set_timeout(2.0)

    world = client.get_world()

    spawn_point_player = world.get_map().get_spawn_points()[48]
    spawn_point_npc = carla.Transform(carla.Location(x=-8.0, y=7.0, z=0.6))

    events, ct, npc_idx = gen_event_data_npc_pedestrian(
        directions,
        t_end,
        dt,
        spawn_point_player,
        spawn_point_npc,
        n_extra_npc_vehicles,
        npc_idx=npc_idx,
        pg_display=pg_display,
        dim=dim,
    )

    if save:
        save_fold = os.path.join(base_fold, "pedestrians", f"example_{index_example}")
        if not os.path.exists(save_fold):
            os.makedirs(save_fold)

        np.save(os.path.join(save_fold, "events" + filename_extension + ".npy"), events)
        np.savez(
            os.path.join(save_fold, "sim_data" + filename_extension + ".npz"),
            collision_time=ct,
            t_end=t_end,
            dt=dt,
            target_vel_kmh=TARGET_VEL_KMH,
            t_cross=t_cross,
        )

    return npc_idx


def run_no_npc(
    t_end,
    dt,
    target_vel_kmh,
    n_extra_npc_vehicles,
    index_example=0,
    save=True,
    pg_display=None,
    dim=(WIDTH, HEIGHT),
    npc_class="pedestrians",
    filename_extension="_baseline",
):
    directions = [
        ("set_autopilot", ["vehicle_player", True], {}, 0.0),
        ("set_desired_speed", ["vehicle_player", target_vel_kmh], {}, 0.0),
    ]

    client = carla.Client("localhost", 2000)
    client.set_timeout(2.0)

    world = client.get_world()

    spawn_point_player = world.get_map().get_spawn_points()[48]

    events, ct = gen_event_data_no_npc(
        directions,
        t_end,
        dt,
        spawn_point_player,
        n_extra_npc_vehicles,
        pg_display=pg_display,
        dim=dim,
    )

    if save:
        save_fold = os.path.join(base_fold, npc_class, f"example_{index_example}")
        if not os.path.exists(save_fold):
            os.makedirs(save_fold)

        np.save(os.path.join(save_fold, "events" + filename_extension + ".npy"), events)
        np.savez(
            os.path.join(save_fold, "sim_data" + filename_extension + ".npz"),
            collision_time=ct,
            t_end=t_end,
            dt=dt,
            target_vel_kmh=TARGET_VEL_KMH,
        )


#pygame.init()
#display = pygame.display.set_mode((WIDTH, HEIGHT))

"""
for vehicle_class, i in product(
    VEHICLE_CLASSES.keys(), range(N_EXAMPLES_PER_VEHICLE_CLASS)
):
    print(f"Running example {i} for vehicle class {vehicle_class}")
    _vehicle_npc_idx = run_car_front_brake(
        T_END,
        T_BRAKE,
        DT,
        TARGET_VEL_KMH,
        MIN_DIST_LEADING_VEHICLE,
        N_EXTRA_NPC_VEHICLES_CAR_FRONT,
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
        N_EXTRA_NPC_VEHICLES_CAR_FRONT,
        vehicle_class,
        index_example=i,
        vehicle_npc_idx=_vehicle_npc_idx,
        save=SAVE,
        # pg_display=display,
    )
"""

for i in range(N_EXAMPLES_PER_VEHICLE_CLASS):
    print(f"Running example {i} for pedestrian crossing")
    _npc_idx = run_pedestrian_crossing(
        T_END,
        T_CROSS,
        DT,
        TARGET_VEL_KMH,
        N_EXTRA_NPC_VEHICLES_PEDESTRIAN_CROSS,
        index_example=i,
        save=SAVE,
        #pg_display=display,
    )

    run_no_npc(
        T_END,
        DT,
        TARGET_VEL_KMH,
        N_EXTRA_NPC_VEHICLES_PEDESTRIAN_CROSS,
        index_example=i,
        save=SAVE,
        #pg_display=display,
        npc_class="pedestrians",
    )

# pygame.quit()
