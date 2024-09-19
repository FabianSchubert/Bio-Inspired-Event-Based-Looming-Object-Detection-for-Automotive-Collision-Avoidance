import numpy as np
import os

from src.config import EVENTS_DTYPE

from src.carla_synth.utils import calc_projected_box_extent

from src.carla_synth.scenario_manager import (
    CrossingScenarioManager,
    NormalDrivingScenarioManager,
)


class EventRecorder:
    def __init__(self, dt, save_fold, width, height, save_data=True):
        self.width = width
        self.height = height

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

    def stop(self, save_data_inst=True):
        if self.recording:
            self.recording = False
            print("stop recording")
            if self.save_data and save_data_inst:
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
                x = (evts_idx % self.width).astype("<u2")
                y = (evts_idx // self.width).astype("<u2")
                st = (np.ones_like(x) * self.t * 1000).astype("<u4")
                pol = (data[evts_idx] == 1).astype("<u2")

                self.event_rec_x = np.concatenate((self.event_rec_x, x))
                self.event_rec_y = np.concatenate((self.event_rec_y, y))
                self.event_rec_p = np.concatenate((self.event_rec_p, pol))
                self.event_rec_t = np.concatenate((self.event_rec_t, st))


class MetaDataRecorder:
    def __init__(self, dt, save_fold, manager, camera, save_data=True):
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

    def stop(self, save_data_inst=True):
        if self.recording:
            self.recording = False
            print("stop recording metadata")
            if self.save_data and save_data_inst:
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

            if isinstance(self.sc_manager, CrossingScenarioManager) and (
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

                box_dims = calc_projected_box_extent(
                    inv_cm_mat, cross_vehicle_box_verts
                )
                # append geometric mean of the box dimensions
                self.av_diam_cross_vehicle.append(np.sqrt(box_dims[0] * box_dims[1]))
            else:
                #print("not recording cross vehicle data")
                #print(self.sc_manager.cross_agent, self.camera)

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
