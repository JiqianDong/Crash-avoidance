from utils import *
from controller import *
import random

class World(object):
    def __init__(self, init_params, env_params):
        """
        init_params: the initialization settings
            ex:
                init_params = dict(
                        city_name = "Town03",
                        cav_loc = 1,
                        speed = 20,
                        bhdv_init_speed = 10,
                        headway = 10,
                        loc_diff = 4.5, # almost crash 
                        headway_2 = 7)
        """
        self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(2.0)

        self.hud = HUD(1700,1000)
        self.world = self.client.load_world(init_params["city_name"])
        
        # make the settings into the synchronous_mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05

        self.world.apply_settings(settings)
        
        self.map = self.world.get_map()
        self.ego_veh = None
        self.LHDV = None
        self.FHDV = None
        self.BHDV = None
        self.lhdv_controlle_type = env_params["lhdv_controlle_type"]
        self.vehicles = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self.radar_sensor = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0

        self.restart(init_params)  
        self.world.on_tick(self.hud.on_world_tick)

    def restart(self, init_params):
        # Set up vehicles
        self.LHDV_FLAGS = random.choice([0,1]) # 0 crash from left, 1 crash from right

        if self.lhdv_controlle_type == "human":
            if self.LHDV_FLAGS == 0: # left
                file_number = random.randint(0, 28)
                path_template = './human_data/left/{}.p'
            else:
                file_number = random.randint(0, 26)
                path_template = './human_data/right/{}.p'
            control_command_file = path_template.format(file_number)

        elif self.lhdv_controlle_type == "default":
            files = ['./control_details/LHDV.p','./control_details/LHDV_right.p'] 
            control_command_file = files[self.LHDV_FLAGS]

        elif self.lhdv_controlle_type == "teleop":
            files = ['./generated_trajs/1.npy']

        else:
            raise NotImplementedError
        
        self.setup_vehicles(**init_params)
        # Set up the sensors.
        self.setup_sensors()
        self.setup_controllers(control_command_file)

    def setup_vehicles(self, 
                       cav_loc,
                       speed,
                       bhdv_init_speed,
                       headway,
                       loc_diff, 
                       headway_2):
        
        lane_width = 3.5
        LHDV_spawn_point = self.world.get_map().get_spawn_points()[cav_loc]
        LHDV_spawn_point.location.y += (self.LHDV_FLAGS*2 - 1)*lane_width 
        # print("LHDV",LHDV_spawn_point.location, "flag", self.LHDV_FLAGS)
        CAV_spawn_point = self.world.get_map().get_spawn_points()[cav_loc]#random.choice(spawn_points) if spawn_points else carla.Transform()
        CAV_spawn_point.location.x -= loc_diff
        # print("CAV",CAV_spawn_point.location)

        FHDV_spawn_point = self.world.get_map().get_spawn_points()[cav_loc]
        FHDV_spawn_point.location.x += headway_2
        FHDV_spawn_point.location.y += (self.LHDV_FLAGS*2 - 1)*lane_width 
        # print("FHDV",FHDV_spawn_point.location)

        BHDV_spawn_point = self.world.get_map().get_spawn_points()[cav_loc]
        BHDV_spawn_point.location.x -= headway

        def get_blueprint(role_name,filters,color):
            blueprint = self.world.get_blueprint_library().filter(filters)[0]
            blueprint.set_attribute('role_name', role_name)
            if blueprint.has_attribute('color'):
                blueprint.set_attribute('color', color)
            return blueprint

        self.ego_veh = self.world.try_spawn_actor(get_blueprint("CAV","model3","204,0,0"), CAV_spawn_point)
        self.LHDV = self.world.try_spawn_actor(get_blueprint("LHDV","tt","255,255,0"), LHDV_spawn_point)
        self.FHDV = self.world.try_spawn_actor(get_blueprint("FHDV",'bmw','128,128,128'), FHDV_spawn_point)
        self.BHDV = self.world.try_spawn_actor(get_blueprint("BHDV",'mustang','51,51,255'), BHDV_spawn_point)

        self.vehicles = [self.ego_veh, self.LHDV, self.FHDV, self.BHDV]

        for i in self.vehicles:
            i.set_target_velocity(carla.Vector3D(x=speed))
        self.BHDV.set_target_velocity(carla.Vector3D(x=bhdv_init_speed))

    def setup_sensors(self):
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0

        self.collision_sensor = CollisionSensor(self.ego_veh, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.ego_veh, self.hud)
        self.gnss_sensor = GnssSensor(self.ego_veh)
        self.imu_sensor = IMUSensor(self.ego_veh)
        self.camera_manager = CameraManager(self.ego_veh, self.hud, 2.2)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.ego_veh)
        self.hud.notification(actor_type)

    # TODO:======== add the teleop controller for lhdv
    def setup_controllers(self, control_command_file):
        self.ego_veh_controller = CAV_controller(self.ego_veh)
        self.lhdv_controller = LHDV_controller(self.LHDV,False,control_command_file)
        # self.bhdv_controller = controller(self.BHDV, True)


    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.ego_veh.get_world().set_weather(preset[0])


    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    # def destroy_sensors(self):
    #     self.camera_manager.sensor.destroy()
    #     self.camera_manager.sensor = None
    #     self.camera_manager.index = None

    def destroy(self):
        if not self.vehicles:
            return 
        if self.radar_sensor is not None:
            self.toggle_radar()
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor] + self.vehicles

        for actor in actors:
            if actor is not None:
                actor.destroy()
        self.vehicles = None
        print("destoyed")