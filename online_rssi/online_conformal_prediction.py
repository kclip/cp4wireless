import torch
import torch.nn as nn
from scipy.io import loadmat, savemat

def rssi2cat(v_rssi): # [-95..-27] -> [0..68]   cat=category
    return v_rssi+95

def cat2rssi(v_cat): #  [0..68] -> [-95..-27]
    return v_cat-95

def get_dataset_Zanella2014(ind_scenario):
    l_dataset_scenario =        ['Aisle','Desk_no_nan','Lab','Outdoor','Room']
    dataset_scenario =          l_dataset_scenario[ind_scenario]
    M =                         loadmat( f'/Zanella2014/{dataset_scenario}.mat')['M'] # from https://ieeexplore.ieee.org/document/6648515?arnumber=6648515
    if ind_scenario in [0,2,3]:
        col_ch_id =             4-1
    if ind_scenario in [1,4]:
        col_ch_id =             3-1
    col_rssi =                  5-1

    return MyDataSet( { 'X':    torch.from_numpy((M[:, col_ch_id] -   18.0 )/ 16.0).view(-1,1).float(),         # normalizing x to be within [-1,+1]
                        'y':    torch.from_numpy((M[:, col_rssi ] - (-50.0))/100.0).view(-1,1).float()    } )   # normalizing y to be within [-1,+1]

def get_dataset_Simmons2022(ind_scenario):
    l_dataset_scenario =        ['P_dBm_out_LOS_Head_Indoor_1khz','P_dBm_out_NLOS_Head_Indoor_1khz','P_dBm_out_LOS_Head_Outdoor_1khz','P_dBm_out_NLOS_Head_Outdoor_1khz']
    dataset_scenario =          l_dataset_scenario[ind_scenario]
    M =                         loadmat( f'/Simmons2022/{dataset_scenario}.mat')['M'] # data set from https://ieeexplore.ieee.org/abstract/document/9795301
    col_rssi =                  0

    return MyDataSet( { 'X':    torch.zeros(M.size).view(-1,1).float(),
                        'y':    torch.from_numpy((M[:, col_rssi ] - (-50.0))/100.0).view(-1,1).float()    } )   # normalizing y to be within [-1,+1] # float! only regression, no classification


class MyDataSet(torch.utils.data.Dataset):
    # Constructor
    def __init__(self, dDataset):  # dDataset['X'] are the inputs and dDataset['y'] (long) are the outputs on purpose
        self.X =                dDataset['X']
        self.y =                dDataset['y']
        self.len =              len(self.y)

    # Getter
    def __getitem__(self, index):
        return self.X[index], self.y[index]

    # Get length
    def __len__(self):
        return self.len

def get_naive_set_predictor(m_logits_y_te,y_te,alpha):
    m_prob_y_te =                       torch.nn.functional.softmax(m_logits_y_te,dim=1)
    m_sorted, m_ind =                   torch.sort(m_prob_y_te,dim=1,descending=True)
    N_te, num_cat =                     m_prob_y_te.shape
    l_set_pred =                        [ [] for _ in range(N_te) ] # list of lists
    v_set_size =                        torch.ones( N_te, dtype=torch.long )
    v_is_in_set =                       torch.zeros(N_te, dtype=torch.bool )
    for i_te in range(N_te):
        v_is_in_set[i_te] =             m_ind[i_te,0]==y_te[i_te,0] # the first label (with highest soft prob) is surly in the predictive set
        cum_prob =                      m_sorted[i_te,0]  # and so does its prob
        while (cum_prob<1-alpha) and (v_set_size[i_te]<num_cat):
            v_set_size[i_te] +=         1
            cum_prob +=                 m_sorted[i_te,v_set_size[i_te]-1]
            v_is_in_set[i_te] |=        m_ind[i_te,v_set_size[i_te]-1]==y_te[i_te,0]
        l_set_pred[i_te] =              m_ind[i_te, : v_set_size[i_te] ]
    ineff =                             v_set_size.float().mean()
    covrg =                             v_is_in_set.float().mean()
    return covrg, ineff, l_set_pred

## Regression

def pinball_loss(yhat, y, tau): # tau is clipped to reside in [0,1]
    e =             y-yhat
    tau_clipped =   min( max(tau, 0.0) , 1.0)
    return torch.max( tau_clipped * e, - (1-tau_clipped) * e)

class mlp_lstm_mlp_reg(nn.Module): # f1: MLP = multi layer perceptron --> f2: LSTM --> f3: MLP
    def __init__(   self,
                    dim_x =             2,
                    f1_hidden_layers =  [32],   # list of hidden layers for f1, excluding input of dim_x+1 and output of f2_input_size
                    f2_input_size =     32,     # dim of w, f1's output, also f2's input of the first layer's cell
                    f2_hidden_size =    64,     # size of one LSTM cell hidden vector (same goes for cell vector)
                    f2_num_layers =     2,      # num of layers
                    f3_hidden_layers =  [32]):  # list of hidden layers for f3, excluding input of f2_hidden_size*f2_num_layers+dim_X and output of 1):
                    #scaling_params = [18.0, 16.0, -60.0, 80.0]   ):  # x_center, x_scale, y_center, y_scale
        super(mlp_lstm_mlp_reg, self).__init__()
        self.dim_x =                dim_x
        ######### for f1 (mlp, preprocessing before lstm) ##########
        v_f1_layers =               [dim_x + 1] + f1_hidden_layers + [f2_input_size]
        self.num_total_layers_f1 =  len(v_f1_layers)-1
        for ind_layer in range(self.num_total_layers_f1):
            cur_fc_name =          'f1_mlp' + str(ind_layer)
            setattr(self, cur_fc_name, nn.Linear(  in_features =    int(v_f1_layers[ind_layer]),
                                                    out_features =   int(v_f1_layers[ind_layer+1]),
                                                    bias=True)  )
        self.activ_f1 =             nn.ELU()
        ######### for f2 (lstm) ##########
        self.f2_lstm =              nn.LSTM(        input_size =     f2_input_size,
                                                    hidden_size =    f2_hidden_size,
                                                    num_layers =     f2_num_layers      )
        ######### for f3 (mlp, postprocessing to get confidence for each class) ##########
        self.f3_mlp =               {}
        v_f3_layers =               [f2_num_layers*f2_hidden_size + dim_x] + f3_hidden_layers + [1]
        self.num_total_layers_f3 =  len(v_f3_layers)-1
        for ind_layer in range(self.num_total_layers_f3):
            cur_fc_name =          'f3_mlp' + str(ind_layer)
            setattr(self, cur_fc_name, nn.Linear(  in_features =    int(v_f3_layers[ind_layer]),
                                                    out_features =   int(v_f3_layers[ind_layer+1]),
                                                    bias=True)  )
        self.activ_f3 =             nn.ELU()

    def forward(self,
                X_seq,              # dim = [seq_len, x_dim] previous K samples' inputs
                y_seq,              # dim = [seq_len, 1    ] previous K samples' labels
                X_cur               # dim = [1      , x_dim] curent imput sample
                ):                  # returns logits of num_cat, without the torch.nn.functional.softmax
        ## f1 ##
        if self.dim_x>0:
            flow =                  torch.cat( (X_seq,y_seq.float()),dim=1 ) # f1's inputs (batch for all sequence)
        else:
            flow =                  y_seq.float() # f1's inputs (batch for all sequence)
        for ind_layer in range(self.num_total_layers_f1):
            cur_fc_name =          'f1_mlp' + str(ind_layer)
            flow =                  self.activ_f1(getattr(self,cur_fc_name)(flow)) # linear with activation
        ## f2 ##
        out, (flow, c_final) =   self.f2_lstm(flow.unsqueeze(dim=1) ) # no need for c_final. flow=h_final, with dim =  [f2_num_layers_lstm,1,f2_hidden_size]
        ## f3 ##
        if self.dim_x > 0:
            flow =                  torch.cat((flow.view(1,-1),X_cur.view(1,-1)),dim=1) # flatten all f2_num_layers into one vector, then concat with x_cur
        else:
            flow =                  flow.view(1,-1) # flatten all f2_num_layers into one vector
        for ind_layer in range(self.num_total_layers_f3):
            cur_fc_name =          'f3_mlp' + str(ind_layer)
            flow =                  getattr(self,cur_fc_name)(flow)
            if (ind_layer < self.num_total_layers_f3-1): # apply activation to all but last
                flow =              self.activ_f3(flow)
        return flow # returns a vector of logits with dim=num_cat

def stretching_func_exponential(theta): # Feldman's varphi in 3.3.1
    lin_th =    0.0
    scaler =    1.0
    if theta>lin_th:
        return scaler*(torch.exp(theta)-1)
    elif theta<-lin_th:
        return scaler *( - (torch.exp(-theta)-1) )
    else:
        return scaler * theta


def stretching_func_quantile(theta): # Feldman's varphi in 3.3.2
    return torch.clip(-theta,0.0,1.0)