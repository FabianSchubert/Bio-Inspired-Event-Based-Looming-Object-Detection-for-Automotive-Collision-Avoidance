import carla
import math
import random
import time
import queue
import numpy as np
import pygame
import os


def calc_steering_angle(target_location, vehicle, prop_gain=0.1):
    # calculate the steering angle
    vehicle_transform = vehicle.get_transform()
    target_vector = np.array(
        [
            target_location.x - vehicle_transform.location.x,
            target_location.y - vehicle_transform.location.y,
        ]
    )
    target_vector = target_vector / np.linalg.norm(target_vector)
    right_vector = vehicle_transform.get_right_vector()
    right_vector = np.array([right_vector.x, right_vector.y])
    right_vector = right_vector / np.linalg.norm(right_vector)

    # calculate the angle between the vectors
    dot = np.dot(target_vector, right_vector)

    return prop_gain * math.asin(dot) * 180 / math.pi


from src.carla_synth.utils import (
    convert_events,
    downsample_events,
    draw_image,
    pol_evt_img_to_rgb,
    calc_projected_box_extent,
)
from src.config import EVENTS_DTYPE


def on_collision(event, timer):
    if timer.record:
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

    if USE_PYGAME:
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
        if coll_event is not None:
            if not (coll_event.other_actor.type_id.startswith(("vehicle", "walker"))):
                print("Collision with non-agent detected, skipping")
                return
            print(coll_event.actor, coll_event.other_actor)
            self.coll_event = coll_event
        else:
            self.coll_event = coll_event
        self.t1 = self.t
        self.record = False
        self.await_finish = True


###### PYGAME ########################

USE_PYGAME = True

###### SIM SETTINGS ########################
DT = 0.01
DT_MS = int(DT * 1000)
FPS = int(1.0 / DT)

WIDTH, HEIGHT = 640, 480

WIDTH_RGB, HEIGHT_RGB = WIDTH * 1, HEIGHT * 1

POS_CAM = carla.Location(x=2.5, y=0.0, z=1.0)
ROT_CAM = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)

DVS_REFR_TIME_NS = 0.001e9
DVS_THRESHOLD = 0.2
DVS_LOG_EPS = 1e-1

TARGET_SPEED_COLL_MPS = 10.0 / 3.6

TIME_TO_COLL = 2.5
SPAWN_DIST_CAR = TARGET_SPEED_COLL_MPS * TIME_TO_COLL  # 20.0
SPAWN_DIST_PED = TARGET_SPEED_COLL_MPS * TIME_TO_COLL  # 20.0

R_RANDOM_OFFSET_MAX_CAR = 0.5
R_RANDOM_OFFSET_MAX_PED = 0.0

T_WAIT = 3.0
T_MAX_SHOW = 4.0
T_CUTOFF_START = 0.05

MOVE_VEHICLES = False
##############################

##### DATA RECORDING SETTINGS ####
SAVE_EXAMPLES = True

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

POS_CAM.x = vehicle.bounding_box.extent.x + 0.05


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

if USE_PYGAME:
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
traffic_manager.vehicle_percentage_speed_difference(
    vehicle,
    (1.0 - TARGET_SPEED_COLL_MPS * 3.6 / 30.0)
    * 100.0,  # calculate the percentage difference between the speed limit (I think it is 30 km/h) and the target speed
)
if not os.path.exists(base_fold):
    os.makedirs(base_fold)
files_in_fold = os.listdir(base_fold)
index_example = len(files_in_fold)

while True:
    timer.step()

    if USE_PYGAME and should_quit(pygame.event.get()):
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
    if USE_PYGAME:
        draw_image(
            display,
            pol_evt_img_to_rgb(events_binned.reshape((HEIGHT, WIDTH))),
            scale=(WIDTH_RGB, HEIGHT_RGB),
        )

    # get vehicle speed
    print(f"{vehicle.get_velocity().length() * 3.6:.2f} km/h", end="\r")

    if USE_PYGAME:
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

        # spawn_type = "pedestrians"
        # spawn_type = "cars"
        spawn_type = np.random.choice(["cars", "pedestrians", "none"])
        # add spawn type?
        # spawn_type = np.random.choice(["pedestrians"])
        # spawn_type = np.random.choice(["cars"])
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
        right_vec = vehicle_transf.get_right_vector()

        steer_angle = vehicle.get_wheel_steer_angle(
            carla.VehicleWheelLocation.Front_Wheel
        )

        turn_rad = (
            vehicle.get_velocity().length()
            * 180.0
            / (np.pi * vehicle.get_angular_velocity().z)
        )
        phi = (SPAWN_DIST_CAR if (spawn_type == "cars") else SPAWN_DIST_PED) / turn_rad

        targ_pos_rel = (
            fwd_vec * np.sin(phi) * turn_rad + right_vec * (1 - np.cos(phi)) * turn_rad
        )

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
            + targ_pos_rel
            + carla.Location(z=5.0)
            + random_offset,
            vehicle_transf.rotation,
        )

        # random rotation on z axis
        agent_transf.rotation.yaw += np.random.rand() * 360.0

        # vehicle.set_autopilot(False)

        path_target = vehicle_transf.location + targ_pos_rel + random_offset

        path_target.z = vehicle_transf.location.z

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

    if timer.record and (spawn_type != "none") and agent is None:
        agent = world.try_spawn_actor(agent_bp, agent_transf)

    if (
        timer.record
        and (spawn_type != "none")
        and (agent is not None)
        and (not ground_fixed)
    ):
        agent_loc = agent.get_location()
        if agent_loc != carla.Location(x=0, y=0, z=0):
            agent_loc.z = (
                agent.bounding_box.extent.z - agent.bounding_box.location.z + 0.05
            )
            agent.set_location(agent_loc)
            if spawn_type == "cars" and MOVE_VEHICLES: # make the car drive
                agent.set_autopilot(True)
            elif spawn_type == "pedestrians":
                # make the walker walk
                agent.apply_control(carla.WalkerControl(speed=1.0))

            print("Ground fixed")

            ground_fixed = True

            vehicle.set_autopilot(False)
            const_speed = vehicle.get_velocity().length()

    if timer.record:
        if (spawn_type != "none") and ground_fixed:
            # vehicle_control = carla.VehicleAckermannControl(
            #    steer=calc_steering_angle(agent.get_transform().location, vehicle, prop_gain=0.2),
            #    steer_speed=0.0,
            #    speed=10.,#const_speed,
            # )
            vehicle_control = carla.VehicleControl(
                steer=calc_steering_angle(
                    agent.get_transform().location, vehicle, prop_gain=0.05
                ),
                throttle=1.0
                if vehicle.get_velocity().length() < TARGET_SPEED_COLL_MPS
                else 0.0,
                brake=0.2
                if vehicle.get_velocity().length() > TARGET_SPEED_COLL_MPS
                else 0.0,
            )
            #print(vehicle.get_velocity().length() < TARGET_SPEED_COLL_MPS)
            #print(vehicle_control.throttle)
            vehicle.apply_control(vehicle_control)
            # vehicle.apply_ackermann_control(vehicle_control)
            # print(const_speed)

            # vehicle_control = vehicle.get_control()

            # vehicle_control.throttle = 0.5
            # vehicle_control.brake = 0.0
            # vehicle_control.hand_brake = False
            # vehicle_control.speed = const_speed
            #

            # traffic_manager.set_path(vehicle, [vehicle.get_location(), agent.get_location()])

        # print(traffic_manager.get_all_actions(vehicle))

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
            agent_box_verts = agent.bounding_box.get_world_vertices(
                agent.get_transform()
            )

            box_dims = calc_projected_box_extent(inv_cm_mat, agent_box_verts)
            # append geometric mean of the box dimensions
            average_diameter_obstacle.append(np.sqrt(box_dims[0] * box_dims[1]))

    if timer.await_finish:
        event_rec = np.array(
            list(zip(event_rec_t, event_rec_x, event_rec_y, event_rec_p)),
            dtype=EVENTS_DTYPE,
        )

        # filter out events before t0
        event_rec = event_rec[event_rec["t"] > ((timer.t0 + T_CUTOFF_START) * 1000)]

        # shift the earliest remaining event to t=0
        event_rec["t"] -= int((timer.t0 + T_CUTOFF_START) * 1000)

        # shift the end time accordingly.
        t_end = int((timer.t1 - timer.t0 - T_CUTOFF_START) * 1000)

        print("Recording finished")

        timer.await_finish = False

        _save = True

        if (timer.coll_event is None) and (spawn_type != "none"):
            _save = False
            if agent is not None:
                print(
                    f"{spawn_type} spawned, but no collision detected for {spawn_type}, skipping example."
                )
            else:
                print(f"Spawning of {spawn_type} failed, skipping example.")
        elif (timer.coll_event is not None) and (spawn_type == "none"):
            print("No object spawned, but collision detected, skipping example.")
            _save = False
        elif (timer.coll_event is None) and (spawn_type == "none"):
            print("Baseline example recording finished, saving example.")
            coll_type = "none"
            avg_dim = None
        else:
            print(f"Collision with {spawn_type} detected, saving example.")
            coll_type = spawn_type
            avg_dim = np.mean(average_diameter_obstacle)

        if SAVE_EXAMPLES and _save:
            avg_vel = np.mean(vehicle_velocities)

            save_fold = os.path.join(
                base_fold,
                f"example_{index_example}",
            )
            if not os.path.exists(save_fold):
                os.makedirs(save_fold)

            np.save(os.path.join(save_fold, "events.npy"), event_rec)
            np.savez(
                os.path.join(save_fold, "sim_data.npz"),
                coll_type=coll_type,
                t_end=t_end,
                dt=DT * 1000,
                vel=avg_vel,
                diameter_object=avg_dim,
            )

            index_example += 1

        vehicle.set_autopilot(True)

        del event_rec
        del event_rec_x
        del event_rec_y
        del event_rec_p
        del event_rec_t

        if agent is not None:
            agent.destroy()

exit_sim()
