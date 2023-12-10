import carla
import pickle
import numpy as np

class Controller(object):
    def __init__(self,  actor, carla_pilot=False): # carla pilot is the autopilot defined by carla

        self.vehilcle = actor
        self.carla_pilot = carla_pilot
        
        self.current_control = {'throttle':0,"steer":0,"brake":0}

        self.steering_increment = 0.1
        self.reset()

        if self.carla_pilot:
            self.vehilcle.set_autopilot()
        else:
            self._control = carla.VehicleControl()
        
    def reset(self):
        self.timestep = 0

    def step(self, actions=None):
        self.timestep += 1
        if isinstance(actions,np.ndarray):
            dec = actions.copy()
            dec = self.process(dec)
            self.apply(dec) 
        

    def process(self, dec):
        return dec

    def apply(self, dec):
        # update current booking 
        self.current_control['throttle'] = dec[0]
        self.current_control['steer'] = dec[1]
        self.current_control['brake'] = dec[2]
        
        self._control.throttle = self.current_control['throttle']
        self._control.steer = self.current_control['steer']
        self._control.brake = self.current_control['brake']

        self.vehilcle.apply_control(self._control)

class CAV_controller(Controller):
    def process(self,dec):
        dec -= 1
        if dec[0]==-1:
            throttle = 0
            brake = 1
        elif dec[0] == 0:
            throttle = 0
            brake = 0
        elif dec[0]==1:
            throttle = 1
            brake = 0
        else:
            print(dec)
            raise Exception("no specific throttle brake control command")
        steer = self.current_control['steer'] + dec[1]*self.steering_increment
        return [throttle, steer, brake]

class LHDVController(Controller):
    def __init__(self, actor, carla_pilot=False, command_file='./control_details/LHDV.p'):
        super().__init__(actor, carla_pilot)
        with open(command_file,'rb') as f:
            self.command_list = pickle.load(f)

    def process(self, dec):
        throttle = dec['throttle']
        steer = dec['steer']
        brake = dec['brake']
        return [throttle, steer, brake]

    def step(self,actions=None):
        self.timestep += 1
        if self.timestep < len(self.command_list) - 1:
            actions = self.command_list[self.timestep]
            dec = self.process(actions)
            self.apply(dec)
        else:
            return 
        
class Teleop_controller(Controller):
    def __init__(self, actor, carla_pilot=False, trajectory_file='./generated_trajs/1.npy'):
        super().__init__(actor, carla_pilot)
        np.load(trajectory_file)



