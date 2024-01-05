from utils import *
from collections import defaultdict
import copy
import pygame

class ENV(object):
    def __init__(self, env_params, init_params, sim_params):
        """
        Base environment for Carla RL
        """
        self.init_params = init_params
        self.env_params = env_params
        self.sim_params = sim_params

        self.set_up_world()
        
        self.ego_veh = None
        self.vehicles = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self.radar_sensor = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self.timestep = 0

        self.restart()  
        self.world.on_tick(self.hud.on_world_tick)


    def set_up_world(self):
        self.client = carla.Client(self.env_params["ip"], self.env_params["port"])
        self.client.set_timeout(2.0)

        self.hud = HUD(1700,1000)
        self.world = self.client.load_world(self.env_params["city_name"])
        
        # make the settings into the synchronous_mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.sim_params["fixed_delta_seconds"]
        self.world.apply_settings(settings)
        
        self.map = self.world.get_map()

    def restart(self):
        # reset the render display panel
        if self.sim_params["render_pygame"]:
            self.display = pygame.display.set_mode((1280,760),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.initialize_vehicles()
        self.setup_sensors()
        self.setup_controllers()
        self.setup_initial_speed()
        self.frame_num = None

    def setup_initial_speed(self):
        raise NotImplementedError

    def initialize_vehicles(self):
        raise NotImplementedError

    def setup_sensors(self):
        """
        Currently only on the ego vehicle
        """
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

    def setup_controllers(self):
        raise NotImplementedError

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.ego_veh.get_world().set_weather(preset[0])


    def tick(self, clock):
        self.hud.tick(self, clock)

    def _render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

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

    def check_collision(self):
        if len(self.collision_sensor.history)>0:
            return self.collision_sensor.history[-1]
        else:
            return None
        
    def render_frame(self):
        if self.display:
            self._render(self.display)
            pygame.display.flip()
        else:
            raise Exception("No display to render")
        

    def carla_update(self):
        """
        roll the simulator
        """
        frame = self.world.tick() # update in the simulator
        # snap_shot = self.world.wait_for_tick() # palse the simulator until future tick used in carla 0.9.5
        if self.frame_num is not None:
            if frame != self.frame_num + 1:
                print('frame skip!')
        self.frame_num = frame

    def reset(self):
        self.destroy()
        self.restart()

        # for the initial steps
        self.current_state = defaultdict(list)  #reinitialize the current states  
        self.timestep = 0
        self.frame_num = None

        warming_up_steps = self.init_params["warming_up_steps"]
        window_size = self.env_params["state_params"]["window_size"]
        assert warming_up_steps>window_size, "warming_up_steps should be larger than the window size"
        self.carla_update()
        [self.step(None) for _ in range(warming_up_steps)]

        return copy.deepcopy(self.current_state)