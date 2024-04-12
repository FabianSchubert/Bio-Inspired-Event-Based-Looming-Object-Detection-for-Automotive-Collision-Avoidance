"""
Simulate a scenario
"""

import os

import carla

import numpy as np

from src.carla_synth.gen_event_data_scenario import gen_event_data_npc_car

from time import time_ns

base_fold = os.path.join(
    os.path.dirname(__file__), "../../../data/carla_sim/car_front_brake/"
)


np.random.seed(time_ns() % 2**32)

T_BRAKE = 10.0
T_END = 15.0
DT = 0.02
T_CUTOFF = 10.0

TARGET_VEL_KMH = 15.0  # km/h
MIN_DIST_LEADING_VEHICLE = 10.0  # meters

SAVE = False

N_EXTRA_NPC_VEHICLES = 25

VEHICLE_INDICES = {
    "cars": [24, 40, 39],
    "two_wheel": [33, 34, 9],
    "trucks": [28, 13, 31],
}


def run_car_front_brake(
    t_end,
    t_brake,
    dt,
    target_vel_kmh,
    min_dist_leading_vehicle,
    n_extra_npc_vehicles,
    vehicle_npc_idx,
    vehicle_class,
    index_example=0,
    save=True,
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

    events, ct = gen_event_data_npc_car(
        directions,
        t_end,
        dt,
        spawn_point_player,
        spawn_point_npc,
        n_extra_npc_vehicles,
        vehicle_npc_idx=vehicle_npc_idx,
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

#'''
for vehicle_class, indices in VEHICLE_INDICES.items():
    for i, vehicle_idx in enumerate(indices):
        print(f"Running example {i} for vehicle class {vehicle_class}")
        run_car_front_brake(
            T_END,
            T_BRAKE,
            DT,
            TARGET_VEL_KMH,
            MIN_DIST_LEADING_VEHICLE,
            N_EXTRA_NPC_VEHICLES,
            vehicle_idx,
            vehicle_class,
            index_example=i,
            save=SAVE,
        )
#'''
'''
i = 2
vehicle_idx = 39
vehicle_class = "cars"
print(f"Running example {i} for vehicle class {vehicle_class}")
run_car_front_brake(
    T_END,
    T_BRAKE,
    DT,
    TARGET_VEL_KMH,
    MIN_DIST_LEADING_VEHICLE,
    N_EXTRA_NPC_VEHICLES,
    vehicle_idx,
    vehicle_class,
    index_example=i,
    save=SAVE,
)
'''
