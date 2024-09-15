import pdb
import carla
import numpy as np
import pygame

import os

from src.carla_synth.utils import (
    convert_events,
    downsample_events,
    pol_evt_img_to_rgb,
    calc_projected_box_extent,
)

from src.config import EVENTS_DTYPE

from itertools import cycle


class ScenarioManager:
    def __init__(
        self,
        world,
        target_vel_vehicle_kmh=20.0,
        target_vel_walker_kmh=5.0,
        dt=0.01,
        waittime=3.0,
    ):
        self.world = world

        self.mp = self.world.get_map()
        self.bpl = self.world.get_blueprint_library()

        self.sptf = self.mp.get_spawn_points()

        ###### hand picked spawn points plus offsets #####
        self.spawn_ids = [0, 2, 3, 11]

        # self.spawn_cycle = cycle(self.spawn_ids)

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
            self.sptf[idx].location += (
                self.sptf[idx].get_forward_vector() * self.offsets_fwd[idx]
            )
            self.sptf[idx].location += (
                self.sptf[idx].get_right_vector() * self.offsets_right[idx]
            )

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
        self.ego = self.world.spawn_actor(self.bpl.find("vehicle.audi.a2"), spawn_tf)

    def destroy_ego(self):
        self.ego.destroy()

    def destroy_cross_agent(self):
        if self.cross_agent is not None:
            self.cross_agent.destroy()
            self.cross_agent = None

    def create_scenario(self, go_right=None, spawn_id=None):
        self.t = 0.0

        self.start_signal_sent = False

        if spawn_id is None:
            spawn_id = np.random.choice(self.spawn_ids)

        if go_right is None:
            go_right = np.random.choice([True, False])

        self.spawn_tf_ego = self.sptf[spawn_id]

        self.ego.set_transform(self.spawn_tf_ego)

        self.destroy_cross_agent()

        self.cross_type = np.random.choice(["walker", "vehicle"])

        dist_fwd = 8.0 if (self.cross_type == "vehicle") else 6.0
        self.dist_horiz = 15.0 if (self.cross_type == "vehicle") else 4.0

        self.th_horiz = 12.0 if (self.cross_type == "vehicle") else 5.0

        self.dir = go_right * 2 - 1

        spawn_pos = (
            self.spawn_tf_ego.location
            + self.spawn_tf_ego.get_forward_vector() * dist_fwd
            - self.spawn_tf_ego.get_right_vector() * self.dist_horiz * self.dir
        )

        spawn_rot = carla.Rotation(
            self.spawn_tf_ego.rotation.pitch,
            self.spawn_tf_ego.rotation.yaw,
            self.spawn_tf_ego.rotation.roll,
        )

        spawn_rot.yaw += 90.0 * self.dir

        spawn_tf = carla.Transform(location=spawn_pos, rotation=spawn_rot)

        bp_veh_crossing = np.random.choice(
            blueprint_library.filter(f"{self.cross_type}.*")
        )
        print(bp_veh_crossing)

        try:
            self.cross_agent = world.spawn_actor(bp_veh_crossing, spawn_tf)
            print("spawned\n==================================")

        except Exception as e:
            print(e)
            self.cross_agent = None
            print("not spawned")
            self.create_scenario()

    def update(self):
        self.t += self.dt

        if self.t >= self.waittime:
            if not self.start_signal_sent:
                self.event_recorder.start()
                self.metadata_recorder.start("none_with_crossing")
                self.start_signal_sent = True

            if self.cross_type == "vehicle":
                self.update_vehicle_control()
            else:
                self.update_walker_control()

                # check if the vehicle has crossed the threshold opposite to where it started

            self.metadata_recorder.fetch_data()

            self.check_threshold()

    def update_vehicle_control(self):
        if self.cross_agent is not None:
            thr = np.maximum(
                0.0,
                15.0
                * (self.target_vel_vehicle - self.cross_agent.get_velocity().length())
                / self.target_vel_vehicle,
            )

            vehicle_control = carla.VehicleControl(throttle=thr, brake=0.0)
            # print(vehicle.get_velocity().length() < TARGET_SPEED_COLL_MPS)
            # print(vehicle_control.throttle)
            self.cross_agent.apply_control(vehicle_control)

    def update_walker_control(self):
        if self.cross_agent is not None:
            walker_control = carla.WalkerControl(
                direction=self.ego.get_transform().get_right_vector() * self.dir,
                speed=self.target_vel_walker,
            )

            self.cross_agent.apply_control(walker_control)

    def check_threshold(self):
        if self.cross_agent is not None:
            diff = (
                self.cross_agent.get_transform().location
                - self.ego.get_transform().location
            )

            proj = self.ego.get_transform().get_right_vector().dot(diff)

            if (self.dir == 1 and proj > self.th_horiz) or (
                self.dir == -1 and proj < -self.th_horiz
            ):
                self.event_recorder.stop()
                self.metadata_recorder.stop()
                self.destroy_cross_agent()
                self.create_scenario()

    def set_event_recorder(self, recorder):
        self.event_recorder = recorder

    def set_metadata_recorder(self, recorder):
        self.metadata_recorder = recorder


class CameraManager:
    def __init__(self, width, height, world, car, sensortype="rgb"):
        self.width = width
        self.height = height

        self.world = world

        self.bpl = self.world.get_blueprint_library()

        self.sensortype = sensortype

        assert sensortype in ["rgb", "dvs"]

        if sensortype == "rgb":
            cam_bp = self.bpl.find("sensor.camera.rgb")

        else:
            cam_bp = self.bpl.find("sensor.camera.dvs")

        self.event_data = None

        cam_bp.set_attribute("image_size_x", str(width))
        cam_bp.set_attribute("image_size_y", str(height))

        pos_cam = carla.Location(x=2.5, y=0.0, z=1.0)
        rot_cam = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)

        self.camera = self.world.spawn_actor(
            cam_bp, carla.Transform(pos_cam, rot_cam), attach_to=car
        )

        self.rgb_arr = np.zeros((height, width, 3), dtype=np.uint8)

        self.surface = pygame.surfarray.make_surface(self.rgb_arr.swapaxes(0, 1))

        if sensortype == "rgb":
            self.camera.listen(self.udpate_callback_rgb)
        else:
            self.camera.listen(self.udpate_callback_dvs)

        self.recorder = None

    def udpate_callback_rgb(self, data):
        img = np.reshape(np.copy(data.raw_data), (self.height, self.width, 4))
        img = img[:, :, :3]
        img = img[:, :, ::-1]
        self.rgb_arr[:] = img
        self.surface = pygame.surfarray.make_surface(self.rgb_arr.swapaxes(0, 1))

        if self.recorder is not None:
            self.recorder.receive_data(data)

    def udpate_callback_dvs(self, data):
        evt_arr = convert_events(data)
        events_binned = downsample_events(evt_arr, self.width, self.height)

        self.rgb_arr[:] = pol_evt_img_to_rgb(
            events_binned.reshape((self.height, self.width))
        )

        self.surface = pygame.surfarray.make_surface(self.rgb_arr.swapaxes(0, 1))

        if self.recorder is not None:
            self.recorder.receive_data(events_binned)

    def destroy(self):
        self.camera.destroy()

    def set_recorder(self, recorder):
        self.recorder = recorder


class EventRecorder:
    def __init__(self, dt, save_fold, save_data=True):
        self.event_rec_x = np.array([], dtype="<u2")
        self.event_rec_y = np.array([], dtype="<u2")
        self.event_rec_p = np.array([], dtype="<u2")
        self.event_rec_t = np.array([], dtype="<u4")

        self.recording = False

        self.dt = dt
        self.t = 0.0

        self.save_mod = SaveModule(save_fold, "events")

        self.save_data = save_data

    def start(self):
        if not self.recording:
            self.t = 0.0
            self.recording = True

            self.event_rec_x = np.array([], dtype="<u2")
            self.event_rec_y = np.array([], dtype="<u2")
            self.event_rec_p = np.array([], dtype="<u2")
            self.event_rec_t = np.array([], dtype="<u4")

            print("start recording")
        else:
            print(
                "already recording! (you need to stop the recording first to restart)"
            )

    def stop(self):
        if self.recording:
            self.recording = False
            print("stop recording")
            if self.save_data:
                print("saving events...")
                event_rec = np.array(
                    list(
                        zip(
                            self.event_rec_t,
                            self.event_rec_x,
                            self.event_rec_y,
                            self.event_rec_p,
                        )
                    ),
                    dtype=EVENTS_DTYPE,
                )

                self.save_mod.save_npy(event_rec)
        else:
            print("already stopped recording!")

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


class MetaDataRecorder:
    def __init__(
        self, dt, save_fold, manager, camera, save_data=True
    ):
        self.ego_velocities = []
        self.av_diam_cross_vehicle = []
        self.coll_type = None

        self.sc_manager = manager
        self.camera = camera

        self.recording = False

        self.save_mod = SaveModule(save_fold, "sim_data")

        self.save_data = save_data

        self.dt = dt
        self.t = 0.0

    def start(self, coll_type):
        if not self.recording:
            self.ego_velocities = []
            self.av_diam_cross_vehicle = []

            self.coll_type = coll_type

            self.recording = True

            self.t = 0.0

            print("start recording metadata")
        else:
            print(
                "already recording metadata! (you need to stop the recording first to restart)"
            )

    def stop(self):
        if self.recording:
            self.recording = False
            print("stop recording metadata")
            if self.save_data:
                print("saving metadata...")

                avg_dim = np.mean(self.av_diam_cross_vehicle)
                avg_vel = np.mean(self.ego_velocities)

                self.save_mod.save_npz(
                    {
                        "coll_type": self.coll_type,
                        "t_end": self.t * 1000,
                        "dt": self.dt * 1000,
                        "vel": avg_vel,
                        "diameter_object": avg_dim,
                    }
                )
        else:
            print("already stopped recording metadata!")

    def fetch_data(self):
        if self.recording:
            self.t += self.dt

            if self.sc_manager.ego is not None and self.sc_manager.ego.is_alive:
                self.ego_velocities.append(self.sc_manager.ego.get_velocity().length())
            else:
                print("warning: ego vehicle not set or destroyed")
                self.ego_velocities.append(np.nan)

            if (
                self.sc_manager.cross_agent is not None
                and self.sc_manager.cross_agent.is_alive
                and self.camera is not None
                and self.camera.is_alive
            ):
                inv_cm_mat = self.camera.get_transform().get_inverse_matrix()
                cross_vehicle_box_verts = (
                    self.sc_manager.cross_agent.bounding_box.get_world_vertices(
                        self.sc_manager.cross_agent.get_transform()
                    )
                )

                box_dims = calc_projected_box_extent(inv_cm_mat, cross_vehicle_box_verts)
                # append geometric mean of the box dimensions
                self.av_diam_cross_vehicle.append(np.sqrt(box_dims[0] * box_dims[1]))
            else:
                print("warning: cross agent or camera not set or destroyed")
                print(self.sc_manager.cross_agent, self.camera)
                
                self.av_diam_cross_vehicle.append(np.nan)


class SaveModule:
    def __init__(self, save_fold, filename):
        self.save_fold = save_fold

        self.filename = filename

        if not os.path.exists(save_fold):
            os.makedirs(save_fold)
        files_in_fold = os.listdir(save_fold)
        # get the index of the next example to prevent overwriting previous examples
        self.index_example = len(files_in_fold)

    def save_npy(self, data):
        save_subfold = os.path.join(
            self.save_fold,
            f"example_{self.index_example}",
        )
        if not os.path.exists(save_subfold):
            os.makedirs(save_subfold)

        np.save(os.path.join(save_subfold, self.filename), data)
        self.index_example += 1

    def save_npz(self, data_dict):
        save_subfold = os.path.join(
            self.save_fold,
            f"example_{self.index_example}",
        )
        if not os.path.exists(save_subfold):
            os.makedirs(save_subfold)

        np.savez(os.path.join(save_subfold, self.filename), **data_dict)
        self.index_example += 1


SAVE_EXAMPLES = True

base_fold = os.path.join(
    os.path.dirname(__file__), "../../../data/carla_sim/random_spawn/"
)

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

event_recorder = EventRecorder(DT, "data_testfold")

metadata_recorder = MetaDataRecorder(
    DT, "data_testfold", scenario_mng, cam_mg.camera
)

cam_dvs_mg.set_recorder(event_recorder)
scenario_mng.set_event_recorder(event_recorder)
scenario_mng.set_metadata_recorder(metadata_recorder)


############################

pygame.init()

display = pygame.display.set_mode((WIDTH, HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)

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
