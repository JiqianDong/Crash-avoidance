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

# try:
#     sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass

import carla
import pygame
import numpy as np

import gym
from gym.spaces.box import Box
from gym.spaces import Discrete, Tuple
from sklearn.metrics.pairwise import euclidean_distances
from collections import defaultdict
from utils import World, HUD
import copy

class CarlaEnv(object):
    '''
        An OpenAI Gym Environment for CARLA.
    '''
    def __init__(self,
                 host='127.0.0.1',
                 port=2000,
                 city_name='Town03',
                 render_pygame=True,
                 warming_up_steps=50,
                 window_size = 5,
                 init_params = None,
                 use_real_human = False
                 ):
        self.init_params = init_params
        if not init_params:
            if city_name == "Town03": cav_loc = 1
            elif city_name == "Town04":cav_loc = 0
            else:
                cav_loc = 0
                print("unknow spawn point") 
            print("using default initial parameters")
            self.init_params = dict(cav_loc = 1,
                                    speed = 20,
                                    bhdv_init_speed = 10,
                                    headway = 10,
                                    loc_diff = 4.5, # almost crash 
                                    headway_2 = 7)

        self.client = carla.Client(host,port)
        self.client.set_timeout(2.0)

        self.hud = HUD(1700,1000)
        self._carla_world = self.client.load_world(city_name)
        
        settings = self._carla_world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05

        self._carla_world.apply_settings(settings)

        self.world = World(self._carla_world, self.hud, self.init_params, use_real_human)
        self.render_pygame = render_pygame

        self.timestep = 0
        self.warming_up_steps = warming_up_steps
        self.window_size = window_size
        self.current_state = defaultdict(list)  #  {"CAV":[window_size, num_features=9], "LHDV":[window_size, num_features=6]}


    @staticmethod
    def action_space(self):
        throttle_brake = Discrete(3)  # -1 brake, 0 keep, 1 throttle
        steering_increment = Discrete(3)
        return Tuple([throttle_brake, steering_increment])

    @staticmethod
    def state_space(self):
        N = len(self.world.vehicles)
        F = 6 # FIXME not hard code
        return Box(low=-np.inf, high=np.inf, shape=(N,F), dtype=np.float32)

    def reset(self):
        # reset the render display panel
        if self.render_pygame:
            self.display = pygame.display.set_mode((1280,760),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.world.destroy()
        self.world.restart(self.init_params)
        self.current_state = defaultdict(list)  #reinitialize the current states
        
        self.timestep = 0
        self.frame_num = None

        self.carla_update()
        assert self.warming_up_steps>self.window_size, "warming_up_steps should be larger than the window size"
        [self.step(None) for _ in range(self.warming_up_steps)]

        return copy.deepcopy(self.current_state)

    def carla_update(self):
        frame = self._carla_world.tick() # update in the simulator
        # snap_shot = self._carla_world.wait_for_tick() # palse the simulator until future tick used in carla 0.9.5
        if self.frame_num is not None:
            # print(self.frame_num)
            if frame != self.frame_num + 1:
                print('frame skip!')
        self.frame_num = frame

    def step(self,rl_actions):
        
        self.world.cav_controller.step(rl_actions)
        self.world.ldhv_controller.step()
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
        if self.render_pygame:
            self.render_frame()

        return state_, reward_, done_, infos

    def render_frame(self):
        if self.display:
            self.world.render(self.display)
            pygame.display.flip()
        else:
            raise Exception("No display to render")


    def check_collision(self):
        if len(self.world.collision_sensor.history)>0:
            return self.world.collision_sensor.history[-1]
        else:
            return None

    def get_state(self):
        '''
        N * [x,y,vx,vy,ax,ay]
        '''
        for veh in self.world.vehicles:
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

        
        current_control = self.world.cav_controller.current_control
        current_control = [current_control['throttle'],current_control['steer'],current_control['brake']]

        ####  one timestep behind

        # current_control = self.world.CAV.get_control()
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



