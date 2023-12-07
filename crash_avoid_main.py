import os
import sys
import glob

try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

from path_planning import path_planning_main
from gather_data import gather_data_main
from train_model import training_main

def main():
    ### GLOBAL PARAMETERS  
    MAX_STEPS_PER_EPISODE = 100
    WARMING_UP_STEPS = 50
    WINDOW_SIZE = 5
    RETURN_SEQUENCE = False
    RENDER = True
    CITY_NAME = "Town03"

    ### FLOW CONTROL FLAGS
    GATHER_DATA = False
    TRAINING = False
    TESTING = True
    RETRAINING = False


    if GATHER_DATA:
        data_path = './experience_data/{}_data_pickle.pickle'.format(CITY_NAME)
        gather_data_main(city_name=CITY_NAME,
                         num_runs=50,
                         warming_up_steps=WARMING_UP_STEPS,
                         window_size=WINDOW_SIZE,
                         render=RENDER,
                         max_steps_per_episode=MAX_STEPS_PER_EPISODE,
                         data_saving_path=data_path)

    if TRAINING:
        data_path = './experience_data/{}_data_pickle.pickle'.format(CITY_NAME)
        models = ['mlp','rnn','linreg']
        # models = ['linreg']
        # models = ['rnn']
        for md in models:
            training_main(model_type = md,
                          num_training_epochs=1000,
                          batch_size=40,
                          data_loading_path = data_path,
                          return_sequence=RETURN_SEQUENCE,
                          loading_pretrained=False)

    if TESTING:
        saving_data = True if RETRAINING else False
        path_planning_main( num_runs=50,
                            max_steps_per_episode=MAX_STEPS_PER_EPISODE, 
                            model_type='linreg', 
                            return_sequence=False, 
                            warming_up_steps=WARMING_UP_STEPS, 
                            window_size=WINDOW_SIZE, 
                            planning_horizon=15, 
                            num_trajectories=10,
                            CEM_iters=5,
                            render=True,
                            saving_data=saving_data)

    if RETRAINING:
        models = ['mlp','rnn','linreg']
        # models = ['linreg']
        # models = ['rnn']
        for model_type in models:
            training_main(model_type,
                          num_training_epochs=1000,
                          batch_size=40,
                          return_sequence=RETURN_SEQUENCE,
                          loading_pretrained=True)


if __name__ == '__main__':
    main()
    
    
    