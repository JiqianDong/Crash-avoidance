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


def MPC_select_action(cav_predictor,hdv_predictor,current_state,hdmap, planning_horizon,num_trajectories=5):

    random_actions = np.random.choice(3,size=[planning_horizon,num_trajectories,2]) # size=[planning_horizon,num_trajectories,2]
    # Give heuristic for braking:
    # random_actions[:,:,0] = 0
    # random_actions[:,:,1] = 1
    current_control = current_state['current_control'][-1]  #[throttle, steering, brake]

    cav_state = torch.tensor(current_state['CAV']).float()
    cav_control = torch.tensor(current_state['current_control']).float()
    cav_X = torch.cat([cav_state,cav_control],dim=-1) 
    cav_X = torch.stack([cav_X]*num_trajectories)  # [num_trj, seq_len, feature_size=9]  

    hdv_X = torch.tensor([current_state[key] for key in current_state \
                    if key!='CAV'and key!='current_control']).float() # [num_hdvs, seq_len, feature_size=6]
    costs = np.zeros(num_trajectories)

    # cost for summation of steering angle
    steering_costs = np.zeros([num_trajectories,planning_horizon])

    gamma = 0.99
    for i in range(planning_horizon):

        current_controls = cav_X[:,-1,-3:] #num_traj,3
 
        current_controls = torch.tensor([process_action(a,c) for a,c in zip(random_actions[i],current_controls)]).float() # [num_trj, 3]
        
        steering_costs[:,i] = current_controls[:,1] # [num_trj, 1]

        cav_output = cav_predictor.forward(cav_X).detach() # [num_trj, feature_size=6] 
        hdv_output = hdv_predictor.forward(hdv_X).detach() # [num_hdvs, feature_size=6]
        
        # compute cost for each trajectory
        costs += gamma**(i+1)*compute_cost(cav_output.clone(), hdv_output.clone(), hdmap)
        hdv_X_new = torch.stack([torch.cat([val[1:,:],new_val.unsqueeze(0)],dim=0) for val,new_val in zip(hdv_X,hdv_output)]).float()
        cav_new_state = torch.cat([cav_output,current_controls],dim=1).unsqueeze(1)
        cav_X_new = torch.cat([cav_X[:,1:,:],cav_new_state],dim=1)

        cav_X = cav_X_new
        hdv_X = hdv_X_new

    # steering_costs = np.abs(np.sum(steering_costs,axis=1)) #[num_traj, 1]
    steering_costs = 0
    best_action = random_actions[0,np.argmin(costs+steering_costs),:]
    return best_action


def optimization_based_action_selection(cav_predictor,hdv_predictor,current_state,planning_horizon):

    pass


def updata_dataset(dataset):
    with open('./experience_data/data_pickle.pickle','rb') as f:
        init_dataset = pickle.load(f)

    init_dataset.append(dataset)
    
    with open('./experience_data/data_pickle.pickle','wb') as f:
        pickle.dump(init_dataset,f,pickle.HIGHEST_PROTOCOL)
        

def avoid_crash(env, num_runs,max_steps_per_episode, cav_predictor,\
                hdv_predictor,planning_horizon,num_trajectories):
    
    dataset = Dataset()

    clock = pygame.time.Clock()
    quit_flag = False
    success_runs = 0

    for episode_num in range(num_runs):
        current_state = env.reset().copy()
        episode_reward = 0
        current_action = np.array([1,1])# !!!! The simulator is 1 time step slower than action 
        # print(current_state)
        for timestep in range(max_steps_per_episode):
            clock.tick()
            env.world.tick(clock)
            # check quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_flag = True

            rl_actions = MPC_select_action(cav_predictor,\
                    hdv_predictor,current_state,env.world.map,planning_horizon,num_trajectories)            
            
            next_state, reward, done, info = env.step(rl_actions) #state: {"CAV":[window_size, num_features=9], "LHDV":[window_size, num_features=6]}
    
            episode_reward += reward
            dataset.add(current_state,current_action,next_state,reward,done)
            current_state = next_state
            current_action = rl_actions

            if done:
                # print(next_state['CAV'])
                # print(info)
                if info["collide_with"] == "Audi Tt":
                    success_runs -= 1
                break

            if quit_flag:
                print("stop in the middle ... ")
                return dataset
        success_runs += 1
        print("Episode ",episode_num," done in : ", timestep, " -- episode reward: ", episode_reward)

    return dataset,success_runs


def path_planning_main(num_runs,max_steps_per_episode, model_type, \
    return_sequence, warming_up_steps, window_size, planning_horizon,num_trajectories, \
    city_name="Town03", render=True, saving_data=True,init_params=None, use_real_human=False):

    ### load the model
    from traj_pred_models import build_model
    seq_flag = 'seq' if return_sequence else 'nonseq'
    cav_model_file = './models/{}/{}_{}_{}.pt'.format(city_name, 'cav', model_type, seq_flag)
    hdv_model_file = './models/{}/{}_{}_{}.pt'.format(city_name, 'hdv', model_type, seq_flag)

    cav_predictor, hdv_predictor, _, _ = build_model(model_type, return_sequence)
    # print(cav_model_file,hdv_model_file)
    try:
        cav_predictor.load_state_dict(torch.load(cav_model_file))
        hdv_predictor.load_state_dict(torch.load(hdv_model_file))

        # print(cav_predictor.output_layer.weight)
        # print(hdv_predictor.output_layer.weight)
        cav_predictor.eval()
        hdv_predictor.eval()

        print("successfully loaded the %s model"%model_type)
    except Exception as e:
        print(e)
        print("no trained model found")
        return 

    ### set up the environment
    env = None
    try:
        pygame.init()
        pygame.font.init()

        # create environment
        env = CarlaEnv( city_name=city_name,
                        render_pygame=render,
                        warming_up_steps=warming_up_steps, 
                        window_size=window_size,
                        init_params = init_params,
                        use_real_human=use_real_human)
        
        dataset,success_runs = avoid_crash( env=env,
                                            num_runs=num_runs,
                                            max_steps_per_episode=max_steps_per_episode,
                                            cav_predictor=cav_predictor,
                                            hdv_predictor=hdv_predictor,
                                            planning_horizon=planning_horizon,
                                            num_trajectories = num_trajectories)
        
        if saving_data:
            updata_dataset(dataset)
        
        print("success_rate: ", success_runs/num_runs)

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

    return success_runs/num_runs


if __name__ == "__main__":
# 
    # RENDER = True
    RENDER = False
    MAX_STEPS_PER_EPISODE = 100
    WARMING_UP_STEPS = 50
    WINDOW_SIZE = 5

    # PLANNING_HORIZONs = [3]
    # NUM_TRAJECORIESs = [30]

    PLANNING_HORIZONs = [1,3,5,7,10]
    NUM_TRAJECORIESs = [5,10,20,30]
    SAVING_DATA = False
    USE_REAL_HUMAN = True
    CITY_NAME = "Town03"
    speed = 25
    init_params = dict(cav_loc = 1,
                       speed = speed,
                       bhdv_init_speed = speed,
                       headway = 10,
                       loc_diff = 4.5, # almost crash 
                       headway_2 = 7)
    # print(init_params)

    # info = {}
    for PLANNING_HORIZON in PLANNING_HORIZONs:
        for NUM_TRAJECORIES in NUM_TRAJECORIESs:    
            sr = path_planning_main(num_runs=20,
                                    max_steps_per_episode=MAX_STEPS_PER_EPISODE, 
                                    model_type='linreg', 
                                    return_sequence=False, 
                                    warming_up_steps=WARMING_UP_STEPS, 
                                    window_size=WINDOW_SIZE, 
                                    planning_horizon=PLANNING_HORIZON, 
                                    num_trajectories = NUM_TRAJECORIES,
                                    city_name=CITY_NAME,
                                    render=RENDER,
                                    saving_data=SAVING_DATA,
                                    init_params=init_params,
                                    use_real_human=USE_REAL_HUMAN)
    
            print("PLANNING_HORIZON: ", PLANNING_HORIZON)
            print("NUM_TRAJECORIES: ", NUM_TRAJECORIES)
            print("Success rate: ", sr)

            # info[]info.get()

            with open("result25_human.txt",'a+') as f:
                s =  str(PLANNING_HORIZON) + " " + str(NUM_TRAJECORIES) + " " + str(sr)
                f.write(s + '\n')