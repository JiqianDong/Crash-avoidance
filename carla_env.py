#!/usr/bin/env python

# Copyright (c) 2019 Intel Labs
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Jiqian's tests

"""

import time
import os
import numpy as np
import sys
import glob
import pygame

try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import numpy as np
import random

from gym.spaces.box import Box
from gym.spaces import Discrete, Tuple
from sklearn.metrics.pairwise import euclidean_distances
from collections import defaultdict
import copy
from controller import *

from env import ENV

class CarlaEnv(ENV):
    '''
        An OpenAI Gym Environment for CARLA.
    '''
    def __init__(self, env_params, init_params, sim_params):
        super().__init__(env_params, init_params, sim_params)

        self.current_state = defaultdict(list)  #  {"CAV":[window_size, num_features=9], "LHDV":[window_size, num_features=6]}
        self.window_size = env_params["state_params"]["window_size"]

    @staticmethod
    def action_space(self):
        throttle_brake = Discrete(3)  # -1 brake, 0 keep, 1 throttle
        steering_increment = Discrete(3)
        return Tuple([throttle_brake, steering_increment])

    @staticmethod
    def state_space(self):
        N = len(self.vehicles)
        F = self.env_params["state_params"]["state_shape"]
        return Box(low=-np.inf, high=np.inf, shape=(N,F), dtype=np.float32)

    def step(self, rl_actions=None):
        
        self.ego_veh_controller.step(rl_actions)
        self.LHDV_controller.step()
        self.carla_update()

        state_ = copy.deepcopy(self.get_state()) #next observation

        collision = self.check_collision()
        done_ = False
        infos = {}
        if collision:
            # print("collision here: ", collision)
            done_ = True
            infos = {"frame":collision[0],
                    "collide_with":collision[2],
                    "intensity":collision[1]}

        reward_ = self.compute_reward(collision)

        self.timestep += 1 
        self.render_frame()

        return state_, reward_, done_, infos


    def get_state(self):
        '''
        N * [x,y,vx,vy,ax,ay]
        '''
        for veh in self.vehicles:
            state = []
            veh_name = veh.attributes['role_name']

            location = veh.get_location()
            state += [location.x, location.y]

            speed = veh.get_velocity()
            state += [speed.x, speed.y]

            accel = veh.get_acceleration()
            state += [accel.x, accel.y]

            if self.current_state and len(self.current_state[veh_name]) == self.window_size:
                self.current_state[veh_name].pop(0)
            self.current_state[veh_name].append(state)

        
        current_control = self.ego_veh_controller.current_control
        current_control = [current_control['throttle'],current_control['steer'],current_control['brake']]
        ####  one timestep behind
        # current_control = self.CAV.get_control()
        # current_control = [current_control.throttle, current_control.steer, current_control.brake]

        if self.current_state and len(self.current_state["current_control"]) == self.window_size:
            self.current_state["current_control"].pop(0)
        self.current_state["current_control"].append(current_control)

        return self.current_state

    def compute_reward(self,collision=None):
        weight_collision = 1
        base_reward = 0
        collision_penalty = 0
        if collision:
            collision_penalty = collision[1] # the negative intensity of collision

        return base_reward - collision_penalty*weight_collision


    # TODO:======== add the teleop controller for lhdv
    def setup_controllers(self):
        LHDV_controlle_type = self.env_params["lhdv_controlle_type"]
        if LHDV_controlle_type == "human_control_command":
            files = ['./control_details/human_simulator/control_command/left_lc.p'
                     './control_details/human_simulator/control_command/right_lc.p']
            control_command_file = files[self.LHDV_LC_DIRECTION]            
            self.LHDV_controller = LHDV_controller(self.LHDV, control_command_file, carla_pilot=False)

        elif LHDV_controlle_type == "default":
            files = ['./control_details/synthetic/LHDV_left.p',
                     './control_details/synthetic/LHDV_right.p'] 
            control_command_file = files[self.LHDV_LC_DIRECTION]
            self.LHDV_controller = LHDV_controller(self.LHDV, control_command_file, carla_pilot=False)

        elif LHDV_controlle_type == "generated_trajs":
            files = ['./generated_trajs/1.npy']


        elif LHDV_controlle_type == "human_trajs":
            files = ['./control_details/human_simulator/trajectories/left_lc.p',
                     './control_details/human_simulator/trajectories/right_lc.p'] 
            traj_file = files[self.LHDV_LC_DIRECTION]

            self.LHDV_controller = Teleop_controller(self.LHDV, traj_file, carla_pilot=False)

        else:
            raise NotImplementedError
        
        self.ego_veh_controller = Ego_controller(self.ego_veh)
        # BHDV_controller = controller(BHDV, True)


    def initialize_vehicles(self):
        
        cav_loc = self.init_params["cav_loc"]
        headway_2 = self.init_params["headway_2"]
        headway = self.init_params["headway"]
        loc_diff = self.init_params["loc_diff"]

        self.LHDV_LC_DIRECTION = random.choice([0,1]) # 0 crash from left, 1 crash from right
        
        lane_width = 3.5
        LHDV_spawn_point = self.world.get_map().get_spawn_points()[cav_loc]
        LHDV_spawn_point.location.y += (self.LHDV_LC_DIRECTION*2 - 1)*lane_width 
        # print("LHDV",LHDV_spawn_point.location, "flag", self.LHDV_LC_DIRECTION)
        CAV_spawn_point = self.world.get_map().get_spawn_points()[cav_loc]#random.choice(spawn_points) if spawn_points else carla.Transform()
        CAV_spawn_point.location.x -= loc_diff
        # print("CAV",CAV_spawn_point.location)

        FHDV_spawn_point = self.world.get_map().get_spawn_points()[cav_loc]
        FHDV_spawn_point.location.x += headway_2
        FHDV_spawn_point.location.y += (self.LHDV_LC_DIRECTION*2 - 1)*lane_width 
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

        self.vehicles = [self.ego_veh, 
                         self.LHDV,
                         self.FHDV, 
                         self.BHDV
                         ]

    def setup_initial_speed(self):
        speed = self.init_params["speed"]
        bhdv_init_speed = self.init_params["bhdv_init_speed"]
        for i in self.vehicles:
            i.set_target_velocity(carla.Vector3D(x=speed))
        self.BHDV.set_target_velocity(carla.Vector3D(x=bhdv_init_speed))

if __name__ == '__main__':
    import yaml
    import traceback

    env = None
    with open("./cfg.yaml", 'r') as f:
        params = yaml.safe_load(f)

    init_params = params["init_params"]
    env_params =  params["env_params"]
    # crash_avoid_params =  params["crash_avoid_params"]
    sim_params =  params["sim_params"]

    try:
        pygame.init()
        pygame.font.init()
        env = CarlaEnv(env_params, init_params, sim_params)

        for _ in range(10):
            current_state = env.reset().copy()
            for _ in range(100):
                next_state, reward, done, info = env.step() 
                if done:
                    print("collision with: ", info["collide_with"])
                    break

    except Exception as e:
        traceback.print_exc()
    finally:

        if env and env.world:       
            env.destroy()           
            settings = env.world.get_settings()
            settings.synchronous_mode = False
            env.world.apply_settings(settings)
            print('\n disabling synchronous mode.')

        pygame.quit()
