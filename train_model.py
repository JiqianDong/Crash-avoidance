import torch
import torch.nn as nn
import json
import pickle

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"

def train_model_step(model,optimizer,loss_fn, x_batch, y_batch):
    model.train()
    # forward pass
    outputs = model.forward(x_batch)
    loss = loss_fn(outputs,y_batch)
    loss_out = loss.item()
    # back propagate
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss_out

def train_one_epoch(cav_model, hdv_model, cav_optimizer, hdv_optimizer, dataset, batch_size, return_sequence):

    cav_model.to(DEVICE)
    cav_loss_fn = nn.MSELoss().to(DEVICE)

    hdv_model.to(DEVICE)
    hdv_loss_fn = nn.MSELoss().to(DEVICE)

    batch_id = 0 
    cav_training_loss = 0
    hdv_training_loss = 0

    for state_batch, action_batch, next_state_batch in dataset.random_iterator(batch_size, return_sequence=return_sequence):
        hdv_X = []
        hdv_Y = []
        for key in state_batch:
            if key!='CAV'and key!='current_control':
                hdv_X.append(state_batch[key])
                hdv_Y.append(next_state_batch[key])
        hdv_X = torch.cat(hdv_X, dim=0).to(DEVICE) #(batch_size*num_vehicles, window_size, feature_size)
        hdv_Y = torch.cat(hdv_Y, dim=0).to(DEVICE) 

        # print(hdv_X.shape, hdv_Y.shape)
        
        # FIXME THERE IS A NEED OF FUSION ACTION WITH STATE
        cav_X = torch.cat([state_batch['CAV'],state_batch['current_control']],dim=-1).to(DEVICE)
        cav_Y = next_state_batch['CAV'].to(DEVICE)

        cav_loss = train_model_step(cav_model,cav_optimizer,cav_loss_fn, cav_X, cav_Y)
        hdv_loss = train_model_step(hdv_model,hdv_optimizer,hdv_loss_fn, hdv_X, hdv_Y)

        cav_training_loss += cav_loss
        hdv_training_loss += hdv_loss

        # print statistics
        if (batch_id+1)%50 == 0:
            print("Batch number {}: ------  cav loss: {} ---- hdv loss: {}".format(batch_id+1,cav_loss, hdv_loss))
        batch_id += 1

    cav_training_loss /= batch_id
    hdv_training_loss /= batch_id

    return cav_training_loss, hdv_training_loss 


def training_main(model_type,num_training_epochs,batch_size,return_sequence=False,loading_pretrained=False):

    #model files
    seq_flag = 'seq' if return_sequence else 'nonseq'
    cav_model_file = './models/{}_{}_{}.pt'.format('cav', model_type, seq_flag)
    hdv_model_file = './models/{}_{}_{}.pt'.format('hdv', model_type, seq_flag)
    
    from traj_pred_models import build_model
    
    # load the saved dataset
    with open('./experience_data/data_pickle.pickle','rb') as f:
        init_dataset = pickle.load(f) 

    if loading_pretrained:
        try:
            cav_predictor.load_state_dict(torch.load(cav_model_file))
            hdv_predictor.load_state_dict(torch.load(hdv_model_file))

            cav_predictor.eval()
            hdv_predictor.eval()

            print("successfully loaded the %s model"%model_type)
        except Exception as e:
            print(e)
            pass
    

    cav_predictor, hdv_predictor, cav_optimizer, hdv_optimizer = build_model(model_type,return_sequence)


    cav_losses = []
    hdv_losses = []

    print("start training %s ... \n"%model_type)
    for i in range(1,num_training_epochs+1):
        if i%50 ==0: print(" ------ start epoch %d ------"%(i))
        cav_training_loss, hdv_training_loss = train_one_epoch(cav_predictor,\
                                                               hdv_predictor,\
                                                               cav_optimizer,\
                                                               hdv_optimizer,\
                                                               init_dataset,\
                                                               batch_size,\
                                                               return_sequence)
        cav_losses.append(cav_training_loss)
        hdv_losses.append(hdv_training_loss)

    
    # save the model
    torch.save(cav_predictor.state_dict(),cav_model_file)
    torch.save(hdv_predictor.state_dict(),hdv_model_file)

    # save the training loss history
    cav_logdir = './training_stats/cav/'
    hdv_logdir = './training_stats/hdv/'

    with open(cav_logdir + model_type +'_training_loss.txt','w') as f:
        json.dump({"training_loss":cav_losses}, f)

    with open(hdv_logdir + model_type +'_training_loss.txt','w') as f:
        json.dump({"training_loss":hdv_losses}, f)
    
    # plot the training curve
    from generate_training_plots import loss_plot
    loss_plot(cav_logdir,'cav')
    loss_plot(hdv_logdir,'hdv')


if __name__ == "__main__":
    models = ['mlp','rnn','linreg']
    # models = ['linreg']
    # models = ['rnn']


    for model_type in models:
        training_main(model_type, 10)
    
