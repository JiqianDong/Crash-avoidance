# Model class
import torch.nn as nn
import torch

def normalize(x, mean, std, eps=1e-8):
    return (x - mean) / (std + eps)

def unnormalize(x, mean, std):
    return x * std + mean



class RNN_Predictor(nn.Module):
    def __init__(self,input_dim, output_dim, encode_dim=None, rnn_hidden_dim=16, return_sequence=False):
        # return sequence defines whether the output is a sequence or just only the last step
        super().__init__()
        if encode_dim:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, encode_dim),
                nn.Linear(encode_dim, encode_dim),
                nn.ELU())
        else:
            self.encoder = nn.Identity()
            encode_dim = input_dim
        self.return_sequence = return_sequence
        self.rnn = nn.GRU(input_size=encode_dim, hidden_size=rnn_hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(in_features=rnn_hidden_dim, out_features=output_dim)

    
    def forward(self, input_batch): 
        # input_batch shape: batch_size , seq_len , feature_size
        batch_size,seq_len,feature_size = input_batch.shape
        embeddings = self.encoder(input_batch) # batch_size , seq_len , encode_dim
        
        # print("embedding shape", embeddings.shape)
        hidden, last = self.rnn(embeddings) # hidden: batch_size , seq_len , hidden_dim; last: 1, batch_size, hidden_dim
        
        if self.return_sequence:
            output = self.output_layer(hidden) # batch_size , seq_len , hidden_dim
        else:
            last = last.view(batch_size,-1) # remove the first "1" in the "last" shape 
            # last = torch.squeeze(last, axis=0) # alternative way of removing extra dimension
            output = self.output_layer(last)
        return output



class MLP_Predictor(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,num_hidden_layers,dropout_prob=0.5,seq_len=5):
        super().__init__()

        self.input_layer = nn.Linear(input_dim*seq_len,hidden_dim)
        self.hidden_layers = nn.ModuleList()

        for i in range(num_hidden_layers):

            self.hidden_layers.append(nn.Linear(hidden_dim,hidden_dim))
            self.hidden_layers.append(nn.Dropout(dropout_prob))
            self.hidden_layers.append(nn.ReLU())

        self.output_layer = nn.Linear(in_features=hidden_dim, out_features=output_dim)
    
    def forward(self, input_batch): 
        # input_batch shape: batch_size, seq_len, feature_size
        batch_size,seq_len,feature_size = input_batch.shape
        flatten = input_batch.view(batch_size,-1) # batch_size, seq_len*feature_size

        x = self.input_layer(flatten) # batch_size, hidden_dim
        
        for layer in self.hidden_layers:
            x = layer(x) # batch_size, hidden_dim

        output = self.output_layer(x) # batch_size * output_dim
        return output



class LinearRegression(nn.Module):
    def __init__(self,input_dim,output_dim,seq_len=5):
        super().__init__()

        self.output_layer = nn.Linear(input_dim*seq_len, output_dim)

    def forward(self, input_batch):
        # input_batch shape: batch_size, seq_len, feature_size
        # print(input_batch.shape)
        batch_size,seq_len,feature_size = input_batch.shape
    
        flatten = input_batch.view(batch_size,-1) # batch_size, seq_len*feature_size
        return self.output_layer(flatten)


def build_model(model_type, return_sequence):
    # setup the model
    if model_type == 'rnn':
        ### MODEL PARAMETERS
        LEARNING_RATE = 0.01
        ENCODE_DIM = 32
        BATCH_SIZE = 15
        
        ### instantiate model
        cav_predictor = RNN_Predictor(input_dim=9, encode_dim=ENCODE_DIM, output_dim=6, return_sequence=return_sequence)
        hdv_predictor = RNN_Predictor(input_dim=6, encode_dim=ENCODE_DIM, output_dim=6, return_sequence=return_sequence)

        ### instantiate optimizer 
        cav_optimizer = torch.optim.Adam(cav_predictor.parameters(),lr=LEARNING_RATE)
        hdv_optimizer = torch.optim.Adam(hdv_predictor.parameters(),lr=LEARNING_RATE)

    elif model_type == 'mlp':
        if return_sequence:
            raise NotImplementedError('mlp is not supporting return sequence')
        ### MODEL PARAMETERS
        LEARNING_RATE = 0.01        
        HIDDEN_DIM = 32
        NUM_HIDDEN_LAYERS = 3
        BATCH_SIZE = 15
        ### instantiate model
        cav_predictor = MLP_Predictor(input_dim=9, hidden_dim=HIDDEN_DIM,output_dim=6,num_hidden_layers=NUM_HIDDEN_LAYERS)
        hdv_predictor = MLP_Predictor(input_dim=6, hidden_dim=HIDDEN_DIM,output_dim=6,num_hidden_layers=NUM_HIDDEN_LAYERS)
        ### instantiate optimizer 
        cav_optimizer = torch.optim.Adam(cav_predictor.parameters(),lr=LEARNING_RATE)
        hdv_optimizer = torch.optim.Adam(hdv_predictor.parameters(),lr=LEARNING_RATE)

    elif model_type == 'linreg':
        if return_sequence:
            raise NotImplementedError('linear regression is not supporting return sequence')
        ### MODEL PARAMETERS
        LEARNING_RATE = 0.01
        BATCH_SIZE = 15  
        ### instantiate model
        cav_predictor = LinearRegression(input_dim=9, output_dim=6)
        hdv_predictor = LinearRegression(input_dim=6, output_dim=6)
        ### instantiate optimizer 
        cav_optimizer = torch.optim.Adam(cav_predictor.parameters(),lr=LEARNING_RATE)
        hdv_optimizer = torch.optim.Adam(hdv_predictor.parameters(),lr=LEARNING_RATE)
    
    else:
        raise NotImplementedError("unknown model class, only rnn, mlp, linreg are supported")


    return cav_predictor, hdv_predictor, cav_optimizer, hdv_optimizer





