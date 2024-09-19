import carla
import numpy as np

from abc import ABC, abstractmethod


class ScenarioManagerBase(ABC):
    """we need the following methods
    - create_scenario()
    - update()
    ...
    THIS IS WIP:
        - find a way to define a generic way how the metadata recorder fetches data from the scenario manager.
        - communication with the event recorder should be easier because the data always has the same format.
    """

    pass


class CrossingScenarioManager:
    def __init__(
        self,
        world,
        target_vel_vehicle_kmh=20.0,
        target_vel_walker_kmh=5.0,
        dt=0.01,
        waittime=3.0,
        max_reps=None,
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

        self.max_reps = max_reps
        self.n_scenarios = 0

        self.finished = False

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

    def destroy_actors(self):
        self.destroy_ego()
        self.destroy_cross_agent()

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

        bp_veh_crossing = np.random.choice(self.bpl.filter(f"{self.cross_type}.*"))
        print(bp_veh_crossing)

        try:
            self.cross_agent = self.world.spawn_actor(bp_veh_crossing, spawn_tf)
            print("spawned\n==================================")
            self.n_scenarios += 1
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
                if self.max_reps is None or self.n_scenarios < self.max_reps:
                    self.create_scenario()
                else:
                    self.finished = True

    def set_event_recorder(self, recorder):
        self.event_recorder = recorder

    def set_metadata_recorder(self, recorder):
        self.metadata_recorder = recorder


##################################


class NormalDrivingScenarioManager:
    def __init__(
        self,
        world,
        traffic_manager,
        n_vehicles=20,
        n_pedestrians=20,
        leading_car_dist=7.5,  # in meters. do not set this too small, otherwise we could get scenarios that "look like a collision".
        dt=0.01,
        # time in between recordings.
        # This is either to allow the vehicles to get going,
        # or to get a bit of change in scenery if respawn_mode is False.
        waittime=15.0,
        sequence_length=3.0, # in seconds. This is the length of the sequence that is recorded.
        max_reps=None, # maximum number of scenarios to be recorded. If None, the scenario manager will keep running until the user stops it.
        respawn_mode=False, # if True, all vehicles are respawned on each new scenario.
    ):
        self.world = world
        self.traffic_manager = traffic_manager

        self.mp = self.world.get_map()
        self.bpl = self.world.get_blueprint_library()

        self.sptf = self.mp.get_spawn_points()

        self.max_reps = max_reps
        self.n_sequences = 0

        self.finished = False

        ############################

        self.respawn_mode = respawn_mode

        self.n_vehicles = n_vehicles
        self.n_pedestrians = n_pedestrians

        self.leading_car_dist = leading_car_dist

        self.vehicles = []
        self.pedestrians = []

        self.spawn_ego(np.random.choice(self.sptf))
        self.ego.set_autopilot(True)
        self.traffic_manager.distance_to_leading_vehicle(
            self.ego, self.leading_car_dist
        )

        self.t = 0.0
        self.dt = dt

        self.waittime = waittime
        self.sequence_length = sequence_length

        self.start_signal_sent = False

        self.create_scenario(init=True)

    def spawn_ego(self, spawn_tf):
        self.ego = self.world.spawn_actor(self.bpl.find("vehicle.audi.a2"), spawn_tf)

    def destroy_ego(self):
        self.ego.destroy()

    def destroy_vehicles(self):
        for vehicle in self.vehicles:
            vehicle.destroy()
            del vehicle
        self.vehicles = []

    def destroy_pedestrians(self):
        for pedestrian in self.pedestrians:
            pedestrian.destroy()
            del pedestrian
        self.pedestrians = []

    def destroy_actors(self):
        self.destroy_ego()
        self.destroy_vehicles()
        self.destroy_pedestrians()

    def create_scenario(self, init=False):
        if self.respawn_mode or init:
            # destroy all vehicles and pedestrians
            self.destroy_vehicles()
            self.destroy_pedestrians()
            self.spawn_tf_ego = np.random.choice(self.sptf)
            self.ego.set_transform(self.spawn_tf_ego)
            while len(self.vehicles) < self.n_vehicles:
                bp_vehicle = np.random.choice(self.bpl.filter("vehicle.*"))
                try:
                    vehicle = self.world.spawn_actor(
                        bp_vehicle, np.random.choice(self.sptf)
                    )
                    vehicle.set_autopilot(True)
                    self.vehicles.append(vehicle)
                except Exception as e:
                    print(e)

            while len(self.pedestrians) < self.n_pedestrians:
                bp_pedestrian = np.random.choice(self.bpl.filter("walker.*"))
                try:
                    pedestrian = self.world.spawn_actor(
                        bp_pedestrian, np.random.choice(self.sptf)
                    )
                    self.pedestrians.append(pedestrian)
                except Exception as e:
                    print(e)

        self.t = 0.0

        self.start_signal_sent = False

    def update(self):
        self.t += self.dt

        if self.t >= self.waittime:
            if not self.start_signal_sent:
                self.event_recorder.start()
                self.metadata_recorder.start("none_with_traffic")
                self.start_signal_sent = True

                # check if the vehicle has crossed the threshold opposite to where it started

            self.metadata_recorder.fetch_data()

        if self.t >= self.sequence_length + self.waittime:
            self.event_recorder.stop()
            self.metadata_recorder.stop()
            if self.max_reps is None or self.n_sequences < self.max_reps:
                self.create_scenario()
                self.n_sequences += 1
            else:
                self.finished = True

    def set_event_recorder(self, recorder):
        self.event_recorder = recorder

    def set_metadata_recorder(self, recorder):
        self.metadata_recorder = recorder
