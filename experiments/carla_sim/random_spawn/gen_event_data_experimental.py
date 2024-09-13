import pdb
import carla
import numpy as np
import pygame
import queue

from src.carla_synth.utils import convert_events, downsample_events, pol_evt_img_to_rgb


class ScenarioManager:

    def __init__(self, world, target_vel_vehicle_kmh=20., target_vel_walker_kmh=5.0, dt=0.01, waittime=3.0):

        self.world = world

        self.mp = self.world.get_map()
        self.bpl = self.world.get_blueprint_library()

        self.sptf = self.mp.get_spawn_points()

        ###### hand picked spawn points plus offsets #####
        self.spawn_ids = [0,2,3,11]

        self.offsets_fwd = {
            0: 5.0,
            2: 5.0,
            3: 8.0,
            11: 13.0,
        }

        self.offsets_right = {
            0: 0.0,
            2: 0.0,
            3: 0.0,
            11: -9.0,
        }

        for idx in self.spawn_ids:
            self.sptf[idx].location += self.sptf[idx].get_forward_vector() * self.offsets_fwd[idx]
            self.sptf[idx].location += self.sptf[idx].get_right_vector() * self.offsets_right[idx]

        ######

        self.cross_agent = None

        self.target_vel_vehicle = target_vel_vehicle_kmh / 3.6
        self.target_vel_walker = target_vel_walker_kmh / 3.6

        self.spawn_ego(self.sptf[0])

        self.t = 0.0
        self.dt = dt

        self.waittime = waittime
        self.start_signal_sent = False

    def spawn_ego(self, spawn_tf):
        self.ego = self.world.spawn_actor(
            self.bpl.find("vehicle.audi.a2"),
            spawn_tf
        )


    def destroy_ego(self):
        self.ego.destroy()

    def destroy_cross_agent(self):
        if self.cross_agent is not None:
            self.cross_agent.destroy()
            self.cross_agent = None

    def create_scenario(self, go_right=True):

        self.t = 0.0

        self.start_signal_sent = False

        spawn_id = np.random.choice(self.spawn_ids)

        self.spawn_tf_ego = self.sptf[spawn_id]

        # shift the position to the ground.
        #new_ego_tf.location.z = (
        #        self.ego.bounding_box.extent.z - self.ego.bounding_box.location.z + 0.005
        #)

        #self.spawn_tf_ego = new_ego_tf

        self.ego.set_transform(self.spawn_tf_ego)


        #_ego_loc = self.ego.get_location()
        #_ego_loc.z = self.ego.bounding_box.extent.z - self.ego.bounding_box.location.z + 0.05

        #self.ego.set_location(carla.Location(_ego_loc))


        self.destroy_cross_agent()

        self.cross_type = np.random.choice(["walker", "vehicle"])

        dist_fwd = 8.0 if (self.cross_type == "vehicle") else 6.0
        self.dist_horiz = 15.0 if (self.cross_type == "vehicle") else 4.0

        self.th_horiz = 12.0 if (self.cross_type == "vehicle") else 5.0

        self.dir = go_right * 2 - 1

        spawn_pos = (self.spawn_tf_ego.location + self.spawn_tf_ego.get_forward_vector() * dist_fwd
                     - self.spawn_tf_ego.get_right_vector() * self.dist_horiz * self.dir)

        spawn_rot = carla.Rotation(self.spawn_tf_ego.rotation.pitch,
                                   self.spawn_tf_ego.rotation.yaw, self.spawn_tf_ego.rotation.roll)

        spawn_rot.yaw += 90. * self.dir

        spawn_tf = carla.Transform(location=spawn_pos, rotation=spawn_rot)

        

        bp_veh_crossing = np.random.choice(blueprint_library.filter(f"{self.cross_type}.*"))
        print(bp_veh_crossing)

        try:

            self.cross_agent = world.spawn_actor(
                bp_veh_crossing,
                spawn_tf
            )
            print("spawned\n==================================")

        except Exception as e:
            print(e)
            self.cross_agent = None
            print("not spawned")
            self.create_scenario(go_right=np.random.choice([True, False]))

    def update(self):

        self.t += self.dt

        if self.t >= self.waittime:

            if not self.start_signal_sent:
                self.recorder.start()
                self.start_signal_sent = True

            if self.cross_type == "vehicle":
                self.update_vehicle_control()
            else:
                self.update_walker_control()

                # check if the vehicle has crossed the threshold opposite to where it started

            self.check_threshold()

    def update_vehicle_control(self):
        if self.cross_agent is not None:
            thr = np.maximum(0.0, 15.0 * (self.target_vel_vehicle -
                                          self.cross_agent.get_velocity().length()) / self.target_vel_vehicle)

            vehicle_control = carla.VehicleControl(
                throttle=thr,
                brake=0.0
            )
            #print(vehicle.get_velocity().length() < TARGET_SPEED_COLL_MPS)
            # print(vehicle_control.throttle)
            self.cross_agent.apply_control(vehicle_control)

    def update_walker_control(self):
        if self.cross_agent is not None:
            walker_control = carla.WalkerControl(
                direction=self.ego.get_transform().get_right_vector() * self.dir,
                speed=self.target_vel_walker
            )

            self.cross_agent.apply_control(walker_control)

    def check_threshold(self):
        if self.cross_agent is not None:
            diff = self.cross_agent.get_transform().location - self.ego.get_transform().location

            proj = self.ego.get_transform().get_right_vector().dot(diff)

            if (self.dir == 1 and proj > self.th_horiz) or (self.dir == -1 and proj < -self.th_horiz):
                self.recorder.stop()
                self.destroy_cross_agent()
                self.create_scenario(go_right=np.random.choice([True, False]))

    def set_recorder(self, recorder):
        self.recorder = recorder


class CameraManager:

    def __init__(self, width, height, world, car, sensortype="rgb"):

        self.width = width
        self.height = height

        self.world = world

        self.bpl = self.world.get_blueprint_library()

        self.sensortype = sensortype

        assert sensortype in ["rgb", "dvs"]

        if sensortype=="rgb":
            cam_bp = self.bpl.find('sensor.camera.rgb')
            
        else:
            cam_bp = self.bpl.find('sensor.camera.dvs')


        self.event_data = None

        cam_bp.set_attribute("image_size_x", str(width))
        cam_bp.set_attribute("image_size_y", str(height))

        pos_cam = carla.Location(x=2.5, y=0.0, z=1.0)
        rot_cam = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)


        self.camera = self.world.spawn_actor(
            cam_bp,
            carla.Transform(pos_cam, rot_cam),
            attach_to=car
            )

        self.rgb_arr = np.zeros((height, width, 3), dtype=np.uint8)

        self.surface = pygame.surfarray.make_surface(self.rgb_arr.swapaxes(0,1))

        if sensortype == "rgb":
            self.camera.listen(self.udpate_callback_rgb)
        else:
            self.camera.listen(self.udpate_callback_dvs)

        self.recorder = None

    def udpate_callback_rgb(self, data):

        img = np.reshape(np.copy(data.raw_data), (self.height, self.width, 4))
        img = img[:,:,:3]
        img = img[:, :, ::-1]
        self.rgb_arr[:] = img
        self.surface = pygame.surfarray.make_surface(self.rgb_arr.swapaxes(0,1))

        if self.recorder is not None:
            self.recorder.receive_data(data)

    def udpate_callback_dvs(self, data):

        evt_arr = convert_events(data)
        events_binned = downsample_events(evt_arr, self.width, self.height)

        self.rgb_arr[:] = pol_evt_img_to_rgb(events_binned.reshape((self.height, self.width)))

        self.surface = pygame.surfarray.make_surface(self.rgb_arr.swapaxes(0,1))
        
        if self.recorder is not None:
            self.recorder.receive_data(events_binned)

    def destroy(self):
        self.camera.destroy()

    def set_recorder(self, recorder):
        self.recorder = recorder

class EventRecorder:

    def __init__(self, dt, save_fold):
        '''
        self.events = np.array(
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
        '''

        self.event_rec_x = np.array([], dtype="<u2")
        self.event_rec_y = np.array([], dtype="<u2")
        self.event_rec_p = np.array([], dtype="<u2")
        self.event_rec_t = np.array([], dtype="<u4")

        self.recording = False

        self.dt = dt
        self.t = 0.0

    def start(self):
        if not self.recording:
            self.t = 0.0
            self.recording = True

            self.event_rec_x = np.array([], dtype="<u2")
            self.event_rec_y = np.array([], dtype="<u2")
            self.event_rec_p = np.array([], dtype="<u2")
            self.event_rec_t = np.array([], dtype="<u4")

            print('start recording')
        else:
            print('already recording! (you need to stop the recording first to restart)')

    def stop(self):
        if self.recording:
            self.recording = False
            print('stop recording')
        else:
            print('already stopped recording!')

    def receive_data(self, data):
        if self.recording:

            self.t += self.dt

            evts_idx = np.where(data != 0)[0]
            if len(evts_idx) > 0:
                x = (evts_idx % WIDTH).astype("<u2")
                y = (evts_idx // WIDTH).astype("<u2")
                st = (np.ones_like(x) * self.t * 1000).astype("<u4")
                pol = (data[evts_idx] == 1).astype("<u2")

                self.event_rec_x = np.concatenate((self.event_rec_x, x))
                self.event_rec_y = np.concatenate((self.event_rec_y, y))
                self.event_rec_p = np.concatenate((self.event_rec_p, pol))
                self.event_rec_t = np.concatenate((self.event_rec_t, st))


WIDTH, HEIGHT = 640, 480

DT = 0.01

POS_CAM = carla.Location(x=2.5, y=0.0, z=1.0)
ROT_CAM = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

world = client.get_world()

_settings = world.get_settings()

world.apply_settings(
    carla.WorldSettings(
        no_rendering_mode=False,
        synchronous_mode=True,
        fixed_delta_seconds=DT,
    )
)

mp = world.get_map()

blueprint_library = world.get_blueprint_library()

traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)


scenario_mng = ScenarioManager(world, dt=DT)


scenario_mng.create_scenario()

cam_mg = CameraManager(WIDTH, HEIGHT, world, scenario_mng.ego)

cam_dvs_mg = CameraManager(WIDTH, HEIGHT, world, scenario_mng.ego, sensortype="dvs")

event_recorder = EventRecorder()

cam_dvs_mg.set_recorder(event_recorder)
scenario_mng.set_recorder(event_recorder)


############################

pygame.init()

display = pygame.display.set_mode(
    (WIDTH, HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF
)

running = True

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                running = False


    display.blit(cam_dvs_mg.surface, (0, 0))

    pygame.display.flip()

    world.tick()

    scenario_mng.update()


pygame.quit()

scenario_mng.destroy_ego()
scenario_mng.destroy_cross_agent()


cam_mg.destroy()
cam_dvs_mg.destroy()

world.apply_settings(_settings)
