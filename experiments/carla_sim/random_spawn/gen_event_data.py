import carla
import math
import random
import time
import queue
import numpy as np
import pygame
import os

from src.carla_synth.utils import (
    convert_events,
    downsample_events,
    draw_image,
    pol_evt_img_to_rgb,
    calc_projected_box_extent,
)
from src.config import EVENTS_DTYPE

from PIL import Image


def on_collision(event, timer):
    timer.end(event)
    print("Collision detected")
    print(event)


def should_quit(events):
    for event in events:
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def exit_sim():
    for actor in actor_list:
        actor.destroy()

    if agent is not None:
        agent.destroy()

    traffic_manager.set_synchronous_mode(False)

    world.apply_settings(_settings)

    pygame.quit()


class CollTiming:
    def __init__(self, dt, t_max):
        self.t0 = 0.0
        self.t1 = 0.0
        self.t = 0.0
        self.dt = dt
        self.record = False
        self.await_finish = False
        self.t_max = t_max
        self.coll_event = None

    def step(self):
        self.t += self.dt
        if self.record and (self.t - self.t0) > self.t_max:
            self.end()

    def start(self):
        self.t0 = self.t
        self.record = True

    def end(self, coll_event=None):
        self.t1 = self.t
        self.record = False
        self.await_finish = True
        if coll_event is not None:
            print(coll_event.actor, coll_event.other_actor)
            self.coll_event = coll_event
        else:
            self.coll_event = coll_event


###### SIM SETTINGS ########################
DT = 0.01
DT_MS = int(DT * 1000)
FPS = int(1.0 / DT)

WIDTH, HEIGHT = 304, 240

WIDTH_RGB, HEIGHT_RGB = 304 * 3, 240 * 3

POS_CAM = carla.Location(x=2.5, y=0.0, z=1.0)
ROT_CAM = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)

DVS_REFR_TIME_NS = 0.001e9
DVS_THRESHOLD = 0.2
DVS_LOG_EPS = 1e-1

SPAWN_DIST_CAR = 15.0
SPAWN_DIST_PED = 15.0

R_RANDOM_OFFSET_MAX_CAR = 0.5
R_RANDOM_OFFSET_MAX_PED = 0.0

T_WAIT = 3.0

T_MAX_SHOW = 4.0
T_CUTOFF_START = 0.5
##############################

##### DATA RECORDING SETTINGS ####
base_fold = os.path.join(
    os.path.dirname(__file__), "../../../data/carla_sim/random_spawn/"
)
#####


## start carla ##
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

world = client.get_world()

mp = world.get_map()

blueprint_library = world.get_blueprint_library()

traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)

_settings = world.get_settings()

world.apply_settings(
    carla.WorldSettings(
        no_rendering_mode=False,
        synchronous_mode=True,
        fixed_delta_seconds=DT,
    )
)

actor_list = []

start_pos = random.choice(mp.get_spawn_points())

vehicle = world.spawn_actor(
    blueprint_library.find("vehicle.audi.a2"),
    start_pos,
)

actor_list.append(vehicle)

agent = None

cam_event_blueprint = blueprint_library.find("sensor.camera.dvs")
cam_event_blueprint.set_attribute("image_size_x", str(WIDTH))
cam_event_blueprint.set_attribute("image_size_y", str(HEIGHT))
cam_event_blueprint.set_attribute("refractory_period_ns", str(DVS_REFR_TIME_NS))
cam_event_blueprint.set_attribute("positive_threshold", str(DVS_THRESHOLD))
cam_event_blueprint.set_attribute("negative_threshold", str(DVS_THRESHOLD))
cam_event_blueprint.set_attribute("log_eps", str(DVS_LOG_EPS))

camera_event = world.spawn_actor(
    cam_event_blueprint,
    carla.Transform(POS_CAM, ROT_CAM),
    attach_to=vehicle,
)

actor_list.append(camera_event)

timer = CollTiming(DT, T_MAX_SHOW)

col_sensor = world.spawn_actor(
    blueprint_library.find("sensor.other.collision"),
    carla.Transform(),
    attach_to=vehicle,
)
actor_list.append(col_sensor)

col_sensor.listen(lambda event: on_collision(event, timer))

event_queue = queue.Queue()
camera_event.listen(event_queue.put)
############

## start pygame ##
pygame.init()

display = pygame.display.set_mode(
    (WIDTH_RGB, HEIGHT_RGB), pygame.HWSURFACE | pygame.DOUBLEBUF
)
###

ground_fixed = False

vehicle.set_autopilot(True)

traffic_manager.ignore_lights_percentage(vehicle, 100)
traffic_manager.ignore_signs_percentage(vehicle, 100)
traffic_manager.ignore_vehicles_percentage(vehicle, 100)
traffic_manager.ignore_walkers_percentage(vehicle, 100)
traffic_manager.distance_to_leading_vehicle(vehicle, 0.0)
# traffic_manager.vehicle_percentage_speed_difference(vehicle, 75)

index_example = {
    "cars": 22,
    "cars_baseline": 20,
    "pedestrians": 17,
    "pedestrians_baseline": 21,
}

while True:
    timer.step()

    if should_quit(pygame.event.get()):
        exit_sim()
        break

    world.tick()

    if not event_queue.empty():
        events_carla = event_queue.get()
        events = convert_events(events_carla)
        del events_carla
    else:
        events = np.array(
            [],
            dtype=np.dtype(
                [
                    ("x", np.uint16),
                    ("y", np.uint16),
                    ("t", np.int64),
                    ("p", bool),
                ]
            ),
        )

    events_binned = downsample_events(events, WIDTH, HEIGHT)
    draw_image(
        display,
        pol_evt_img_to_rgb(events_binned.reshape((HEIGHT, WIDTH))),
        scale=(WIDTH_RGB, HEIGHT_RGB),
    )

    pygame.display.flip()

    if not timer.record and (timer.t - timer.t1) > T_WAIT:
        agent = None
        ground_fixed = False
        timer.start()

        event_rec_x = np.array([], dtype="<u2")
        event_rec_y = np.array([], dtype="<u2")
        event_rec_p = np.array([], dtype="<u2")
        event_rec_t = np.array([], dtype="<u4")

        vehicle_velocities = []
        average_diameter_obstacle = []

        spawn_type = np.random.choice(["cars", "pedestrians", "None"])
        # spawn_type = np.random.choice(["Pedestrian"])
        # spawn_type = np.random.choice(["None"])

        print(f"now showing {spawn_type}")

        # spawn vehicle
        if spawn_type == "cars":
            vh_bps = blueprint_library.filter("vehicle.*")
            car_bps = [
                x for x in vh_bps if int(x.get_attribute("number_of_wheels")) == 4
            ]
            agent_bp = np.random.choice(car_bps)
        elif spawn_type == "pedestrians":
            agent_bp = np.random.choice(blueprint_library.filter("walker.*"))
            if agent_bp.has_attribute("is_invincible"):
                agent_bp.set_attribute("is_invincible", "false")
        else:
            continue

        vehicle_transf = vehicle.get_transform()
        fwd_vec = vehicle_transf.get_forward_vector()

        random_offset_r = np.sqrt(np.random.rand()) * (
            R_RANDOM_OFFSET_MAX_CAR
            if (spawn_type == "cars")
            else R_RANDOM_OFFSET_MAX_PED
        )
        random_offset_angle = np.random.rand() * 2 * np.pi

        random_offset = carla.Location(
            x=random_offset_r * np.cos(random_offset_angle),
            y=random_offset_r * np.sin(random_offset_angle),
            z=0.0,
        )

        agent_transf = carla.Transform(
            vehicle_transf.location
            + (SPAWN_DIST_CAR if (spawn_type == "cars") else SPAWN_DIST_PED) * fwd_vec
            + carla.Location(z=5.0)
            + random_offset,
            vehicle_transf.rotation,
        )

        # random rotation on z axis
        agent_transf.rotation.yaw += np.random.rand() * 360.0

        # vehicle.set_autopilot(False)

        path_target = (
            vehicle_transf.location
            + (SPAWN_DIST_CAR if (spawn_type == "cars") else SPAWN_DIST_PED) * fwd_vec
            + random_offset
        )

        path_target.z = vehicle_transf.location.z

        traffic_manager.set_path(vehicle, [path_target, path_target, path_target])

        # traffic_manager.ignore_lights_percentage(vehicle, 100)
        # traffic_manager.ignore_signs_percentage(vehicle, 100)
        # traffic_manager.ignore_vehicles_percentage(vehicle, 100)
        # traffic_manager.ignore_walkers_percentage(vehicle, 100)

        # import pdb; pdb.set_trace()
        # get the current control and set it on a straight line
        # _vehicle_control = vehicle.get_control()
        # _vehicle_control.throttle = 0.1
        # _vehicle_control.steer = 0.0
        # _vehicle_control.brake = 0.0
        # _vehicle_control.hand_brake = False
        # _vehicle_control.reverse = False

        # vehicle.apply_control(_vehicle_control)

    if timer.record and (spawn_type != "None") and agent is None:
        agent = world.try_spawn_actor(agent_bp, agent_transf)

    if (
        timer.record
        and (spawn_type != "None")
        and (agent is not None)
        and (not ground_fixed)
    ):
        agent_loc = agent.get_location()
        if agent_loc != carla.Location(x=0, y=0, z=0):
            agent_loc.z = (
                agent.bounding_box.extent.z - agent.bounding_box.location.z + 0.05
            )
            agent.set_location(agent_loc)

            print("Ground fixed")

            ground_fixed = True

    if timer.record:
        evts_idx = np.where(events_binned != 0)[0]
        if len(evts_idx) == 0:
            continue
        x = (evts_idx % WIDTH).astype("<u2")
        y = (evts_idx // WIDTH).astype("<u2")
        st = (np.ones_like(x) * timer.t * 1000).astype("<u4")
        pol = (events_binned[evts_idx] == 1).astype("<u2")

        event_rec_x = np.concatenate((event_rec_x, x))
        event_rec_y = np.concatenate((event_rec_y, y))
        event_rec_p = np.concatenate((event_rec_p, pol))
        event_rec_t = np.concatenate((event_rec_t, st))

        vehicle_velocities.append(vehicle.get_velocity().length())

        if agent is not None:
            inv_cm_mat = camera_event.get_transform().get_inverse_matrix()
            agent_box_verts = agent.bounding_box.get_world_vertices()

            box_dims = calc_projected_box_extent(inv_cm_mat, agent_box_verts)
            # append geometric mean of the box dimensions
            average_diameter_obstacle.append(np.sqrt(box_dims[0] * box_dims[1]))

    if timer.await_finish:
        event_rec = np.array(
            list(zip(event_rec_t, event_rec_x, event_rec_y, event_rec_p)),
            dtype=EVENTS_DTYPE,
        )
        event_rec = event_rec[event_rec["t"] > (T_CUTOFF_START + timer.t0) * 1000]

        event_rec["t"] -= int(timer.t0 * 1000)

        t_end = int((timer.t1 - timer.t0) * 1000)

        print("Recording finished")

        timer.await_finish = False

        if timer.coll_event is not None:
            coll_type = spawn_type
            filename_extension = ""

            # get the average diameter of the colliding object
            # projected onto the direction of motion.

        else:
            coll_type = random.choice(["cars", "pedestrians"])
            filename_extension = "_baseline"

        # average velocity over the trial in m/s
        avg_vel = np.mean(vehicle_velocities)
        avg_dim = np.mean(average_diameter_obstacle)

        save_fold = os.path.join(
            base_fold,
            coll_type,
            f"example_{index_example[coll_type + filename_extension]}",
        )
        if not os.path.exists(save_fold):
            os.makedirs(save_fold)

        np.save(os.path.join(save_fold, "events" + filename_extension + ".npy"), events)
        np.savez(
            os.path.join(save_fold, "sim_data" + filename_extension + ".npz"),
            collision_time=t_end,
            t_end=t_end,
            dt=DT * 1000,
            vel=avg_vel,
        )

        np.savez(os.path.join(save_fold, "sim_data" + filename_extension + ".npz"))

        index_example[coll_type + filename_extension] += 1

        del event_rec
        del event_rec_x
        del event_rec_y
        del event_rec_p
        del event_rec_t

        if agent is not None:
            agent.destroy()
