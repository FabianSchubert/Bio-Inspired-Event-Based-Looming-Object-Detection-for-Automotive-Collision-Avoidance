import carla
import numpy as np
import random
import pygame
import queue

from cProfile import Profile
from pstats import SortKey, Stats

from mpl_multiproc import NBPlot

import matplotlib.pyplot as plt

from src.looming_sim.emd.simulator_EMD import EMD_model  # as Simulator
from src.looming_sim.lgmd.simulator_LGMD import LGMD_model  # as Simulator
from src.looming_sim.format_spike_data import tiled_events

from src.looming_sim.emd.network_settings import params as params_emd
from src.looming_sim.lgmd.network_settings import params as params_lgmd


def should_quit(events):
    for event in events:
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def convert_events(events):
    evt_arr = np.frombuffer(
        events.raw_data,
        dtype=np.dtype(
            [
                ("x", np.uint16),
                ("y", np.uint16),
                ("t", np.int64),
                ("p", bool),
            ]
        ),
    ).copy()
    # evt_arr['p'] = evt_arr['p'].astype(int) * 2 - 1
    return evt_arr


def downsample_events(events_array, clip=1):
    idx_x = events_array["x"].astype(int)
    idx_y = events_array["y"].astype(int)
    idx_flat = idx_y * WIDTH + idx_x
    sum_events = np.clip(
        np.bincount(
            idx_flat,
            weights=events_array["p"].astype(int) * 2 - 1,
            minlength=WIDTH * HEIGHT,
        ),
        -clip,
        clip,
    )

    return sum_events  # .reshape((HEIGHT, WIDTH))


def exit_sim():
    pl.plot(0, finished=True)
    plot_output.plot(0, finished=True)
    #pl_hidden.plot(0, finished=True)

    for actor in actor_list:
        actor.destroy()

    world.apply_settings(_settings)

    pygame.quit()


class KeyControl:
    def __init__(self, vehicle):
        self.control = carla.VehicleControl()
        self.vehicle = vehicle

    def parse_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.control.throttle = 1.0
            elif event.key == pygame.K_DOWN:
                self.control.brake = 1.0
            elif event.key == pygame.K_LEFT:
                self.control.steer = -1.0
            elif event.key == pygame.K_RIGHT:
                self.control.steer = 1.0
            elif event.key == pygame.K_r:
                self.control.reverse = not self.control.reverse
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_UP:
                self.control.throttle = 0.0
            elif event.key == pygame.K_DOWN:
                self.control.brake = 0.0
            elif event.key == pygame.K_LEFT:
                self.control.steer = 0.0
            elif event.key == pygame.K_RIGHT:
                self.control.steer = 0.0

    def apply_control(self):
        self.vehicle.apply_control(self.control)

    def update(self, events):
        for event in events:
            self.parse_event(event)
        self.apply_control()


##############################
DT = 0.02
DT_MS = int(DT * 1000)
FPS = int(1.0 / DT)

WIDTH, HEIGHT = 304, 240
WIDTH_RGB, HEIGHT_RGB = 320, 240

pl = NBPlot(mode="image", shape=(HEIGHT, WIDTH), vmin=-1, vmax=1, flip_y=False)
plot_output = NBPlot(mode="line", n_lines=3, vmin=-1, vmax=1)

POS_CAM = carla.Location(x=2.5, y=0.0, z=1.0)
ROT_CAM = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)

DVS_REFR_TIME_NS = 0.001e9
DVS_THRESHOLD = 0.3
DVS_LOG_EPS = 5e-1
##############################

## simulator settings ##
Simulator = EMD_model
params = params_emd

N_SUBDIV = 2
HALF_STRIDE = True

p = params.copy()
p["NT_MAX"] = 10
p["HALF_STEP_TILES"] = HALF_STRIDE
p["N_SUBDIV_X"] = N_SUBDIV
p["N_SUBDIV_Y"] = N_SUBDIV
p["DT_MS"] = DT_MS
N_TILES = (N_SUBDIV * 2 - 1) if HALF_STRIDE else N_SUBDIV

idx_center = N_SUBDIV // 2
idx_tile_center = N_TILES * idx_center + idx_center
###

## start carla ##
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

world = client.get_world()

mp = world.get_map()

blueprint_library = world.get_blueprint_library()

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

cam_rgb_blueprint = blueprint_library.find("sensor.camera.rgb")
cam_rgb_blueprint.set_attribute("image_size_x", str(WIDTH_RGB))
cam_rgb_blueprint.set_attribute("image_size_y", str(HEIGHT_RGB))

camera_rgb = world.spawn_actor(
    cam_rgb_blueprint,
    carla.Transform(POS_CAM, ROT_CAM),
    attach_to=vehicle,
)

actor_list.append(camera_rgb)

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


image_queue = queue.Queue()
camera_rgb.listen(image_queue.put)

event_queue = queue.Queue()
camera_event.listen(event_queue.put)

key_control = KeyControl(vehicle)
###

## start pygame ##
pygame.init()

display = pygame.display.set_mode((WIDTH_RGB, HEIGHT_RGB), pygame.HWSURFACE | pygame.DOUBLEBUF)

clock = pygame.time.Clock()
###

## start simulator ##
simulator = Simulator(p)

#pl_hidden = NBPlot(
#    mode="image",
#    shape=(simulator.S_height, simulator.S_width),
#    vmin=-1e-4,
#    vmax=1e-4,
#    flip_y=False,
#)

def sim_loop():
    while True:
        events = pygame.event.get()

        if should_quit(events):
            break

        world.tick()

        key_control.update(events)

        image = image_queue.get()

        if not event_queue.empty():
            events_carla = event_queue.get()
            events = convert_events(events_carla)
            events_binned = downsample_events(events, 1)

            pl.plot(events_binned.reshape((HEIGHT, WIDTH)))

            evts_idx = np.where(events_binned != 0)[0]
            x = (evts_idx % WIDTH).astype("<u2")
            y = (evts_idx // WIDTH).astype("<u2")
            st = (np.ones_like(x) * DT_MS).astype("<u4")
            pol = (events_binned[evts_idx] == 1).astype("<u2")

            evts_arr = np.array(
                list(zip(st, x, y, pol)),
                dtype=[("t", "<u4"), ("x", "<u2"), ("y", "<u2"), ("p", "<u2")],
            )
            tld_evts = tiled_events(
                evts_arr, WIDTH, HEIGHT, N_SUBDIV, N_SUBDIV, HALF_STRIDE
            )
            x = [tl_data[-1]["x"] for tl_data in tld_evts]
            y = [tl_data[-1]["y"] for tl_data in tld_evts]
            st = [tl_data[-1]["t"] for tl_data in tld_evts]
            pol = [tl_data[-1]["p"] for tl_data in tld_evts]

            spike_bitmasks, pol_bitmasks = simulator.generate_bitmasks(x, y, st, pol)

            # set the simulator's timestep to 0 so that it matches the incoming data
            simulator.model.timestep = 0
            simulator.spk_bm = spike_bitmasks
            simulator.pol_bm = pol_bitmasks

            # push the data to the device
            simulator.push_input_data_to_device()

            simulator.model.step_time()

            simulator.OUT[idx_tile_center].pull_var_from_device("V")
            V = simulator.OUT[idx_tile_center].vars["V"].view[0]

            simulator.OUT[idx_tile_center].pull_var_from_device("r_left")
            r_left = simulator.OUT[idx_tile_center].vars["r_left"].view[0]

            simulator.OUT[idx_tile_center].pull_var_from_device("r_right")
            r_right = simulator.OUT[idx_tile_center].vars["r_right"].view[0]

            #simulator.S[idx_tile_center].pull_var_from_device("vx")
            #vx = np.reshape(
            #    simulator.S[idx_tile_center].vars["vx"].view[:],
            #    (simulator.S_height, simulator.S_width),
            #)

            plot_output.plot(np.array([V, r_left, r_right]))
            #pl_hidden.plot(vx)

        draw_image(display, image)

        pygame.display.flip()

        clock.tick(FPS)
    
    exit_sim()

with Profile() as profile:
    sim_loop()
    print(Stats(profile)
          .strip_dirs()
          .sort_stats(SortKey.TIME)
          .print_stats())