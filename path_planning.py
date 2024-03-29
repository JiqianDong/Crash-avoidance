from platform import dist
import numpy as np
import pygame
import torch
import sys,glob,os
try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
from carla_env import CarlaEnv
import carla
from sklearn.metrics import pairwise_distances
import pickle
from dataset import Dataset
import time
import yaml
import traceback


def process_action(action, current_control):
    # this function is the same as in the controller module
    dec = action.copy()
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

    steer = current_control[1] + dec[1]*0.1
    return [throttle, steer, brake]


def TTC_batch_computing(cav_output, hdv_output, safety_distance):
    # cav_output.shape = [num_trj, feature_size=6] x,y,vx,vy,ax,ay
    # hdv_output.shape = [num_hdvs, feature_size=6] 
    ### TTC function 
    #{||delta_d|| - safety_distance}/{(-delta_v.dot(delta_d)) / ||delta_d||}

    diff = cav_output[:,:4].unsqueeze(1) - hdv_output[:,:4].unsqueeze(0) # [n_traj, n_hdv, 4] (x,y,vx,vy)

    n_traj, n_hdv, _ = diff.shape

    x_diff = diff[:,:,:2] # [n_traj, n_hdv, 2]
    v_diff = diff[:,:,2:] # [n_traj, n_hdv, 2]

    dist = torch.norm(x_diff,p=2,dim=-1) #[n_traj, n_hdv]
    projection = torch.bmm(v_diff.view(-1,1,2),x_diff.view(-1,2,1)).view(n_traj,n_hdv)
    projection = torch.div(-projection,dist)

    valid_distance = dist - safety_distance
    valid_distance[valid_distance<0] = 1e-4 # already below safety distance

    ttc = torch.div(valid_distance,projection) # [n_traj, n_hdv]

    ttc[ttc<0] = float('inf')

    return ttc

def distance_batch_computing(cav_output, hdv_output):
    diff = cav_output[:,:4].unsqueeze(1) - hdv_output[:,:4].unsqueeze(0)
    x_diff = diff[:,:,:2] #[n_traj, n_hdv, 2]
    dist = torch.norm(x_diff,p=2,dim=-1) #[n_traj, n_hdv]
    return dist

def distance_to_road(cav_output):
    '''
        cav_output: [num_traj, 6]
    '''
    d1 = (cav_output[:,1:2] - 0)**2
    d2 = (cav_output[:,1:2] - 15)**2

    d = torch.cat([d1,d2],dim=-1)
    min_d,_ = torch.min(d**0.5,dim=1)

    return min_d

def TTC_to_road(cav_output):
    '''
        cav_output: [num_traj, 6]
    '''
    cav_output = cav_output.numpy()
    left_edge_dist = 15 - cav_output[:,1]
    right_edge_dist = 0 - cav_output[:,1] 

    ttc_left = left_edge_dist / cav_output[:,3]
    ttc_right = right_edge_dist / cav_output[:,3]

    ttc_left[ttc_left<0] = 0
    ttc_right[ttc_right<0] = 0

    return ttc_left + ttc_right

def deviation_from_center(cav_output, hdmap):
    '''
    Compute the deviation from the lane center
    cav_output: [num_traj, 6]
    '''
    distances = []
    for point in np.array(cav_output[:,:2]):
        loc = carla.Location(x = float(point[0]), y = float(point[1]), z=0.133)
        center_loc = hdmap.get_waypoint(loc, project_to_road=True)
        # print(loc)
        # print(center_loc)
        dist = loc.distance(center_loc.transform.location)
        distances.append(dist)
    return np.array(distances)


def compute_cost(cav_state, hdv_state, hdmap):

    ##### Distance cost
    # cav_state = cav_state.numpy()
    # hdv_state = hdv_state.numpy()
    # dist_matrix = pairwise_distances(cav_state[:,:2],hdv_state[:,:2]) #[num_trj, num_hdvs]
    # threshold = 3
    # inv_dist_matrix = 1/dist_matrix
    # # cost = dist_matrix(dist_matrix<threshold).sum(axis=1)
    # cost = np.where(inv_dist_matrix>1/threshold,inv_dist_matrix,0).sum(axis=1) #[num_trj,]
    
    # cost = np.where(dist_matrix<threshold,dist_matrix,0).sum(axis=1) #[num_trj,]

    ###### TTC cost 
    ttc = TTC_batch_computing(cav_state, hdv_state, 2).numpy() # [num_traj, num_hdv]
    ttc[ttc>5] = float('inf')
    ttc_cost = (1/ttc).sum(axis=1)
    
    ### ttc to road edge
    ttc_to_road = TTC_to_road(cav_state)
    ttc_to_road[ttc_to_road==0] = float('inf')  
    ttc_to_road[ttc_to_road>5] = float('inf')
    ttc_to_road_cost = (1/ttc_to_road)
    ### distance to road edge 
    # distance_cost = 1/distance_to_road(cav_state).numpy()

    # print('ttc: ',ttc_cost)
    # # print('dist: ',distance_cost)
    # print('ttc_to_road: ',ttc_to_road_cost)

    # cost = ttc_cost 
    dfc = deviation_from_center(cav_state, hdmap)

    cost = ttc_cost + 0.0 * dfc #+ 0.3*ttc_to_road_cost #*0.5#+distance_cost
    # print("ttc cost",ttc_cost)
    # print("dfc cost", dfc)
    # print()
    return cost

def compute_trajectory_costs(random_action_trajectories, 
                             current_state, cav_predictor, hdv_predictor,
                             hdmap):
    
    planning_horizon, num_trajectories, _ = random_action_trajectories.shape
    cav_state = torch.tensor(current_state['CAV']).float()
    cav_control = torch.tensor(current_state['current_control']).float()
    cav_X = torch.cat([cav_state,cav_control],dim=-1) 
    cav_X = torch.stack([cav_X]*num_trajectories)  # [num_trj, seq_len, feature_size=9]  

    hdv_X = torch.tensor([current_state[key] for key in current_state \
                    if key!='CAV'and key!='current_control']).float() # [num_hdvs, seq_len, feature_size=6]
    costs = np.zeros(num_trajectories)

    # cost for summation of steering angle
    steering_costs = np.zeros([num_trajectories, planning_horizon])
    gamma = 0.99
    for i in range(planning_horizon):

        current_controls = cav_X[:,-1,-3:] #num_traj,3
 
        current_controls = torch.tensor([process_action(a,c) for a,c in zip(random_action_trajectories[i],current_controls)]).float() # [num_trj, 3]
        
        steering_costs[:,i] = np.abs(current_controls[:,1]) # [num_trj, 1] penalize the large steering angles
        # print("steering_min_cost: ", np.min(steering_costs))
        cav_output = cav_predictor.forward(cav_X).detach() # [num_trj, feature_size=6] 
        hdv_output = hdv_predictor.forward(hdv_X).detach() # [num_hdvs, feature_size=6]
        
        # compute cost for each trajectory
        costs += gamma**(i+1) * compute_cost(cav_output.clone(), hdv_output.clone(), hdmap)
        hdv_X_new = torch.stack([torch.cat([val[1:,:],new_val.unsqueeze(0)],dim=0) for val,new_val in zip(hdv_X,hdv_output)]).float()
        cav_new_state = torch.cat([cav_output,current_controls],dim=1).unsqueeze(1)
        cav_X_new = torch.cat([cav_X[:,1:,:],cav_new_state],dim=1)

        cav_X = cav_X_new
        hdv_X = hdv_X_new
    # print(costs, 0.5*steering_costs.sum(axis=1))
    return costs + 0.3 * steering_costs.sum(axis=1)
    

def MPC_select_action(cav_predictor, hdv_predictor, current_state, 
                      hdmap, planning_horizon,num_trajectories,CEM_iters):
    
    K = num_trajectories // 3
    p_throttle = [np.array([1/3, 1/3, 1/3])] * planning_horizon
    p_steering = [np.array([1/3, 1/3, 1/3])] * planning_horizon
    random_throttles = np.array([np.random.choice(np.arange(3), \
            size=num_trajectories, p=p) for p in p_throttle]) # planning_horizon * num_trajectories
    
    random_steerings = np.array([np.random.choice(np.arange(3), \
            size=num_trajectories, p=p) for p in p_steering]) # planning_horizon * num_trajectories
    
    random_action_trajectories = np.dstack([random_throttles, random_steerings]) # size=[planning_horizon,num_trajectories,2] 2:throttle, steering
    
    for i in range(CEM_iters):

        costs = compute_trajectory_costs(random_action_trajectories, 
                                        current_state, cav_predictor, hdv_predictor,
                                        hdmap)
        
        sorted_inds = np.argsort(costs)
        random_action_trajectories = random_action_trajectories[:, sorted_inds, :]

        # refit 
        elite_throttles = random_action_trajectories[:, :K, 0] # planning_horizon * (num_trajectories-K)
        elite_steerings = random_action_trajectories[:, :K, 1] # planning_horizon * (num_trajectories-K)

        p_throttle = []
        p_steering = []
        for h in range(planning_horizon):
            throt_count = dict(zip([0,1,2], [1,1,1]))
            steer_count = dict(zip([0,1,2], [1,1,1]))

            v_throt, p_throt = np.unique(elite_throttles[h,:],  return_counts=True)
            v_steer, p_steer = np.unique(elite_steerings[h,:],  return_counts=True)
            throt_count.update(dict(zip(v_throt, p_throt)))
            steer_count.update(dict(zip(v_steer, p_steer)))

            p_throt = [throt_count[0], throt_count[1], throt_count[2]]
            p_steer = [steer_count[0], steer_count[1], steer_count[2]]

            p_throttle.append(p_throt/np.sum(p_throt))
            p_steering.append(p_steer/np.sum(p_steer))
        

        # Generate new samples and replace the bottom num_trajectories - K
        random_throttles = np.array([np.random.choice(np.arange(3), \
            size=num_trajectories-K, p=p) for p in p_throttle]) # planning_horizon * (num_trajectories-K)
    
        random_steerings = np.array([np.random.choice(np.arange(3), \
                size=num_trajectories-K, p=p) for p in p_steering])  # planning_horizon * (num_trajectories-K)

        random_action_trajectories[:, K:, 0] = random_throttles # planning_horizon * (num_trajectories-K)
        random_action_trajectories[:, K:, 1] = random_steerings # planning_horizon * (num_trajectories-K)


    best_action = random_action_trajectories[0, 0, :]
    return best_action




def updata_dataset(dataset):
    with open('./experience_data/data_pickle.pickle','rb') as f:
        init_dataset = pickle.load(f)

    init_dataset.append(dataset)
    
    with open('./experience_data/data_pickle.pickle','wb') as f:
        pickle.dump(init_dataset,f,pickle.HIGHEST_PROTOCOL)
        

def avoid_crash(env, num_runs,max_steps_per_episode, cav_predictor,\
                hdv_predictor,planning_horizon,num_trajectories,CEM_iters):
    
    dataset = Dataset()

    clock = pygame.time.Clock()
    quit_flag = False
    success_runs = 0
    time_list = []
    for episode_num in range(num_runs):
        current_state = env.reset().copy()
        episode_reward = 0
        current_action = np.array([1,1])# !!!! The simulator is 1 time step slower than action 
        # print(current_state)
        for timestep in range(max_steps_per_episode):
            clock.tick()
            # env.world.tick(clock)
            # check quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_flag = True
            start_time = time.time()
            rl_actions = MPC_select_action(cav_predictor,
                                            hdv_predictor,
                                            current_state,
                                            env.map,
                                            planning_horizon,
                                            num_trajectories,
                                            CEM_iters)            
            end_time = time.time()
            
            time_elapsed = end_time - start_time
            # print(start_time,end_time, time_elapsed)
            time_list.append(time_elapsed)

            next_state, reward, done, info = env.step(rl_actions) #state: {"CAV":[window_size, num_features=9], "LHDV":[window_size, num_features=6]}
    
            episode_reward += reward
            dataset.add(current_state,current_action,next_state,reward,done)
            current_state = next_state
            current_action = rl_actions

            if done:
                # print(next_state['CAV'])
                print("collision with: ", info["collide_with"])
                if info["collide_with"] == "Audi Tt" or \
                    info["collide_with"] == "Mustang Mustang" or \
                        info["collide_with"] == "Tesla Model3" or \
                            info["collide_with"] == "Bmw":
                    success_runs -= 1
                break

            if quit_flag:
                print("stop in the middle ... ")
                return dataset
        success_runs += 1
        print("Episode ",episode_num," done in : ", timestep, " -- episode reward: ", episode_reward)

    return dataset, success_runs, np.mean(time_list)


def path_planning_main(params):

    init_params = params["init_params"]
    env_params =  params["env_params"]
    crash_avoid_params =  params["crash_avoid_params"]
    sim_params =  params["sim_params"]

    ### load the model

    from traj_pred_models import build_model
    seq_flag = 'seq' if crash_avoid_params["return_seq"] else 'nonseq'

    cav_predictor, hdv_predictor, _, _ = build_model(crash_avoid_params["model_type"], 
                                                     crash_avoid_params["return_seq"])
    cav_model_file = './models/{}/{}_{}_{}.pt'.format(env_params["city_name"], 
                                                      'cav', 
                                                      crash_avoid_params["model_type"], 
                                                      seq_flag)
    hdv_model_file = './models/{}/{}_{}_{}.pt'.format(env_params["city_name"], 
                                                      'hdv', 
                                                      crash_avoid_params["model_type"], 
                                                      seq_flag)
    try:
        cav_predictor.load_state_dict(torch.load(cav_model_file))
        hdv_predictor.load_state_dict(torch.load(hdv_model_file))
        cav_predictor.eval()
        hdv_predictor.eval()

        print("successfully loaded the %s model"%crash_avoid_params["model_type"])
    except Exception as e:
        print(e)
        print("no trained model found")
        return 

    ### set up the environment
    env = None
    try:
        pygame.init()
        pygame.font.init()


        env = CarlaEnv(env_params, init_params, sim_params)

        dataset,success_runs,avg_time = avoid_crash( 
                                        env=env,
                                        num_runs=sim_params["num_runs"],
                                        max_steps_per_episode=sim_params["max_steps_per_episode"],
                                        cav_predictor=cav_predictor,
                                        hdv_predictor=hdv_predictor,
                                        planning_horizon=crash_avoid_params["planning_horizon"],
                                        num_trajectories=crash_avoid_params["num_trajectories"],
                                        CEM_iters=crash_avoid_params["CEM_iters"])
        
        if sim_params["saving_data"]:
            updata_dataset(dataset)
        
        print("success_rate: ", success_runs/sim_params["num_runs"])

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

    return success_runs/sim_params["num_runs"], avg_time


if __name__ == "__main__":
    # with open("./configs/default_cfg.yaml", 'r') as f:
    with open("./configs/human_cfg.yaml", 'r') as f:
        params = yaml.safe_load(f)
    sr,avg_t = path_planning_main(params)

    print("Success rate: ", sr)
    print("Average time: ", avg_t)


    # with open(f"./runs/speed_{speed}_success_rate2.txt",'a+') as f:
    #     s =  f"{PLANNING_HORIZON},{NUM_TRAJECORIES},{CEM_ITERS},{sr},{avg_t}"
    #     f.write(s + '\n')