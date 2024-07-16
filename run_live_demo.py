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

from src.carla_synth.utils import draw_image, convert_events, downsample_events


def should_quit(events):
    for event in events:
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False





def exit_sim():
    #pl.plot(0, finished=True)
    #plot_output.plot(0, finished=True)
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
model = "LGMD"

output_scale = 1.0 if model == "EMD" else 1000.0

DT = 0.02
DT_MS = int(DT * 1000)
FPS = int(1.0 / DT)

WIDTH, HEIGHT = 304, 240
WIDTH_RGB, HEIGHT_RGB = 320, 240
WIDTH_SHOW_RGB, HEIGHT_SHOW_RGB = 640, 480

#pl = NBPlot(mode="image", shape=(HEIGHT, WIDTH), vmin=-1, vmax=1, flip_y=False)
#plot_output = NBPlot(mode="line", n_lines=3, vmin=-output_scale, vmax=output_scale)

POS_CAM = carla.Location(x=2.5, y=0.0, z=1.0)
ROT_CAM = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)

DVS_REFR_TIME_NS = 0.001e9
DVS_THRESHOLD = 0.2
DVS_LOG_EPS = 1e-1
##############################

## simulator settings ##
#Simulator = EMD_model
#params = params_emd

Simulator = EMD_model if model == "EMD" else LGMD_model
params = params_emd if model == "EMD" else params_lgmd

N_SUBDIV = 2
HALF_STRIDE = True

p = params.copy()
p["NT_MAX"] = 10
p["HALF_STEP_TILES"] = HALF_STRIDE
p["N_SUBDIV_X"] = N_SUBDIV
p["N_SUBDIV_Y"] = N_SUBDIV
p["DT_MS"] = DT_MS
N_TILES = (N_SUBDIV * 2 - 1) if HALF_STRIDE else N_SUBDIV
p["REC_SPIKES"] = []

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


#image_queue = queue.Queue()


event_queue = queue.Queue()
camera_event.listen(event_queue.put)

key_control = KeyControl(vehicle)
###

## start pygame ##
pygame.init()

display = pygame.display.set_mode((WIDTH_SHOW_RGB, HEIGHT_SHOW_RGB), pygame.HWSURFACE | pygame.DOUBLEBUF)

clock = pygame.time.Clock()
###

camera_rgb.listen(lambda x: draw_image(display, x, scale=(WIDTH_SHOW_RGB, HEIGHT_SHOW_RGB)))

## start simulator ##
#simulator = Simulator(p)

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

        #image = image_queue.get()

        if not event_queue.empty():
            pass
            #events_carla = event_queue.get()
            #events = convert_events(events_carla)
            #events_binned = downsample_events(events, WIDTH, HEIGHT, clip=1)

            #del events
            #del events_carla

            #pl.plot(events_binned.reshape((HEIGHT, WIDTH)))

            '''
            evts_idx = np.where(events_binned != 0)[0]
            x = (evts_idx % WIDTH).astype("<u2")
            y = (evts_idx // WIDTH).astype("<u2")
            st = (np.ones_like(x) * DT_MS).astype("<u4")
            pol = (events_binned[evts_idx] == 1).astype("<u2")

            evts_arr = np.array(
                list(zip(st, x, y, pol)),
                dtype=[("t", "<u4"), ("x", "<u2"), ("y", "<u2"), ("p", "<u2")],
            )

            del x
            del y
            del st
            del pol

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

            del x
            del y
            del st
            del pol

            del evts_arr
            del tld_evts
            del spike_bitmasks
            del pol_bitmasks

            simulator.model.step_time()

            simulator.OUT[idx_tile_center].pull_var_from_device("V")
            V = simulator.OUT[idx_tile_center].vars["V"].view[0]

            if model == "EMD":
                simulator.OUT[idx_tile_center].pull_var_from_device("r_left")
                r_left = simulator.OUT[idx_tile_center].vars["r_left"].view[0]

                simulator.OUT[idx_tile_center].pull_var_from_device("r_right")
                r_right = simulator.OUT[idx_tile_center].vars["r_right"].view[0]
            else:
                simulator.OUT[idx_tile_center].pull_var_from_device("VE")
                r_left = simulator.OUT[idx_tile_center].vars["VE"].view[0]

                simulator.OUT[idx_tile_center].pull_var_from_device("VI")
                r_right = simulator.OUT[idx_tile_center].vars["VI"].view[0]

            #simulator.S[idx_tile_center].pull_var_from_device("vx")
            #vx = np.reshape(
            #    simulator.S[idx_tile_center].vars["vx"].view[:],
            #    (simulator.S_height, simulator.S_width),
            #)

            plot_output.plot(np.array([V, r_left, r_right]))
            #pl_hidden.plot(vx)
            '''

        #draw_image(display, image, scale=(WIDTH_SHOW_RGB, HEIGHT_SHOW_RGB))

        #del image

        pygame.display.flip()

        #clock.tick(FPS)
    
    exit_sim()

with Profile() as profile:
    sim_loop()
    print(Stats(profile)
          .strip_dirs()
          .sort_stats(SortKey.TIME)
          .print_stats())

#sim_loop()