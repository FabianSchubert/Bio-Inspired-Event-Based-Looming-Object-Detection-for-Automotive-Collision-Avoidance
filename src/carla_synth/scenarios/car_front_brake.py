"""
Simulate a scenario where the vehicle is driving in a straight line and
the car in front of it suddenly brakes. The vehicle should detect the
braking car and stop before colliding with it.
"""

import carla

import pygame

from src.carla_synth.utils import CameraManager, HUD

import random

import sys

WIDTH, HEIGHT = 300, 300

DT_SECS = 0.1

N_BG_VEHICLES = 150

pygame.font.init()

client = carla.Client("localhost", 2000)
client.set_timeout(2.0)

actor_list = []

world = client.get_world()

world_settings = world.get_settings()
world_settings.synchronous_mode = True  # Enables synchronous mode
world_settings.fixed_delta_seconds = DT_SECS
world.apply_settings(world_settings)

tm = client.get_trafficmanager()
tm.set_synchronous_mode(True)

blueprint_library = world.get_blueprint_library()

bp_agent = blueprint_library.filter("vehicle")[0]
idx_tf = 48  # random.choice(range(len(world.get_map().get_spawn_points())))
transform_agent = world.get_map().get_spawn_points()[idx_tf]


vehicle_agent = world.spawn_actor(bp_agent, transform_agent)

actor_list.append(vehicle_agent)

hud = HUD(WIDTH, HEIGHT)

camera_manager = CameraManager(vehicle_agent, hud, 2.2)
camera_manager.set_sensor(0, notify=False, force_respawn=True)

bp_car_npc = blueprint_library.filter("vehicle")[0]
transform_car_npc = carla.Transform(transform_agent.location, transform_agent.rotation)
transform_car_npc.location.x -= 6
vehicle_npc = world.spawn_actor(bp_car_npc, transform_car_npc)


actor_list.append(vehicle_npc)

spawn_points = world.get_map().get_spawn_points()
n_spawn_points = len(spawn_points)

if N_BG_VEHICLES < n_spawn_points:
    random.shuffle(spawn_points)
    spawn_points = spawn_points[:N_BG_VEHICLES]
else:
    print("Not enough spawn points for background vehicles")

# Create some background vehicles

for i in range(N_BG_VEHICLES):
    bp_car_bg = random.choice(blueprint_library.filter("vehicle"))
    try:
        vehicle_bg = world.spawn_actor(bp_car_bg, spawn_points[i])
        vehicle_bg.set_autopilot(True)
        actor_list.append(vehicle_bg)
    except Exception as e:
        print(f"Failed to spawn vehicle {i}")

# a list of tuples, each containing a command to execute, the arguments to
# pass to that command, the keyword arguments to pass to that command, and
# the time at which to execute that command.
# Each time the simulation time exceeds the time given for the
# first command in the list, it is executed and removed from the list.
# This means that the commands should be in order of increasing time.
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
]

#### main loop

pygame.init()

display = pygame.display.set_mode((WIDTH, HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)

display.fill((0, 0, 0))

clock = pygame.time.Clock()

t = 0.0

while True:
    # clock.tick_busy_loop(60)
    t += DT_SECS

    #print(f"t: {t}")

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            for actor in actor_list:
                actor.destroy()
            camera_manager.sensor.destroy()
            camera_manager.sensor = None
            camera_manager.index = None

            world_settings.synchronous_mode = False
            world.apply_settings(world_settings)
            tm.set_synchronous_mode(False)

            sys.exit()

    if len(directions) > 0 and t >= directions[0][3]:
        print(f"Executing direction: {directions[0]}")
        directions[0][0](*directions[0][1], **directions[0][2])
        directions.pop(0)

    world.tick()

    camera_manager.render(display)

    pygame.display.flip()
