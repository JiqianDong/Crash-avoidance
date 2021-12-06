import pickle
import pygame
from carla_env import CarlaEnv
from dataset import Dataset
import numpy as np
import time

def gather_data(env, num_runs, max_steps_per_episode, save_info):

    dataset = Dataset()
    clock = pygame.time.Clock()
    quit_flag = False
    for episode_num in range(num_runs):
        current_state = env.reset().copy()
        episode_reward = 0
        current_action = np.array([1,1])# !!!! The simulator is 1 time step slower than action 

        for timestep in range(max_steps_per_episode):
            clock.tick()
            env.world.tick(clock)
            # check quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_flag = True
            rl_actions = np.random.choice(3,2)  # 0 brake, 1 keep, 2 throttle,  steering increment (0,1,2)

            print('RL action',rl_actions)
            next_state, reward, done, _ = env.step(rl_actions) #state: {"CAV":[window_size, num_features=9], "LHDV":[window_size, num_features=6]}
            # print(np.array(state['current_control']).shape) #(5,3) throttle, steering, brake
            print(env.world.CAV.get_control(),'\n')
            
            print(current_state['CAV'][-1], next_state['CAV'][-1], '\n')
            episode_reward += reward
            dataset.add(current_state,current_action,next_state,reward,done)
            current_state = next_state
            current_action = rl_actions

            if done:
                # print(next_state['CAV'])
                break

            if quit_flag:
                print("stop in the middle ... ")
                return

        print("Episode ",episode_num," done in : ", timestep, " -- episode reward: ", episode_reward)

    if save_info:
        with open('./experience_data/data_pickle.pickle','wb') as f:
            pickle.dump(dataset,f,pickle.HIGHEST_PROTOCOL)


def gather_data_main(num_runs,warming_up_steps, window_size,render, max_steps_per_episode, save_info):
    env = None
    try:
        pygame.init()
        pygame.font.init()

        # create environment
        env = CarlaEnv(render_pygame=render,warming_up_steps=warming_up_steps, window_size=window_size)
        gather_data(env, num_runs, max_steps_per_episode, save_info)

    except Exception as e:
        print (e)

    finally:
        if env and env.world:       
            env.world.destroy()           
            settings = env._carla_world.get_settings()
            settings.synchronous_mode = False
            env._carla_world.apply_settings(settings)
            print('\ndisabling synchronous mode.')

        pygame.quit()