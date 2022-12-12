import torch
import numpy as np

def compute_val_ineff(indicator_matrix, Y_te):
    # Y_te: (num_te, 1), indicator_matrix: (num_te, num_classes)
    one_hot_label = torch.nn.functional.one_hot(torch.squeeze(Y_te), num_classes=indicator_matrix.shape[1])
    assert one_hot_label.shape == indicator_matrix.shape
    validity_matrix = one_hot_label * indicator_matrix
    validity = torch.sum(validity_matrix)
    ineff = torch.sum(indicator_matrix)
    return validity, ineff
def NC_compute(prob_vec, y):
    # prob_vec: (num_te, num_classes) # num_te=1 for NC_i
    num_classes = prob_vec.shape[1]
    eps_for_log = 1e-10 # for numerical stability..
    if isinstance(y, int) or len(y.shape) == 0:
        prob_y = prob_vec[:, y] # (num_te)
        return -torch.log(prob_y.unsqueeze(dim=1)+eps_for_log) # (num_te, 1)
    else:
        tmp_y_one_hot = torch.nn.functional.one_hot(torch.squeeze(y), num_classes=num_classes)
        #print(prob_vec.shape, tmp_y_one_hot.shape, tmp_y_one_hot)
        assert prob_vec.shape == tmp_y_one_hot.shape
        prob_y = torch.sum(prob_vec*tmp_y_one_hot, dim=1)
        return -torch.log(prob_y.unsqueeze(dim=1)+eps_for_log)

def compute_prob_vec(var, X, xi, num_classes):
    assert len(X.shape) > 1 # batch dim at 0
    return xi(X, var, if_softmax_out=True) # would be useful for naive set predictor..
def quantile_plus(vec_x, tau): #tau would be 1-alpha
    assert len(vec_x.shape) == 3 # (num_te, N+1, 1)
    N = vec_x.shape[1] # this is actually N+1 in our notations
    sorted_vec_x, _ = torch.sort(vec_x, dim=1)
    ind_quantile = np.ceil((tau)*(N)) - 1 ### python index starts from 0
    return sorted_vec_x[:, int(ind_quantile)]
def hard_indicator(quan_plus):
    # (num_te, 1)
    return (quan_plus >= 0).type(torch.uint8) 
def minimum_of_available_NC(NC_y_dict, y_prime, N, quantile_mode):
    tmp_whole_NC_y_dict_y_prime = []
    for i in range(N):
        tmp_whole_NC_y_dict_y_prime.append(NC_y_dict[i][y_prime]) #[ (num_te,1), ... ]
    tmp_whole_NC_y_dict_y_prime = torch.cat(tmp_whole_NC_y_dict_y_prime, dim=1) # (num_te, N)
    min_NC_y_dict_y_prime, _ = torch.min(tmp_whole_NC_y_dict_y_prime, dim=1, keepdim=True) # (num_te,1)
    return min_NC_y_dict_y_prime
def compute_pred_set_as_indicator_matrix(NC_y_dict, N, alpha, num_classes, if_JK_mm):
    pred_set_as_indicator_matrix = []
    device_info = NC_y_dict[0]['NC_i'].get_device()
    if device_info == -1:
        device = "cpu"
    else:
        device = "cuda:" + str(device_info)
    for y_prime in range(num_classes):
        v_prime = []
        if_debug = True
        if if_JK_mm:
            min_NC_y_dict_y_prime = minimum_of_available_NC(NC_y_dict, y_prime, N, quantile_mode, tau_for_soft_min)
        else:
            pass

        for i in range(N):
            if if_JK_mm:    
                curr_v_prime =  NC_y_dict[i]['NC_i'] - min_NC_y_dict_y_prime # (1,1) - (num_te, 1) -> (num_te, 1)
            else:  
                curr_v_prime =  NC_y_dict[i]['NC_i'] - NC_y_dict[i][y_prime] # (1,1) - (num_te, 1) -> (num_te, 1)
            v_prime.append(curr_v_prime)
        tmp_for_dtype = torch.rand(1)
        max_value = torch.tensor([torch.finfo(tmp_for_dtype.dtype).max]).to(device) # (1)
        max_value = max_value * torch.ones(curr_v_prime.shape).to(device)
        v_prime.append(max_value)
        v_prime = torch.cat(v_prime, dim=1) # [(num_te,1), (num_te,1), ..., (num_te,1)] -> (num_te, N+1)
        v_prime = v_prime.unsqueeze(dim=2) # (num_te, N+1, 1)
        q_plus_v_prime = quantile_plus(v_prime, 1-alpha)
        curr_indicator_column = hard_indicator(q_plus_v_prime) # (num_te, 1)
        pred_set_as_indicator_matrix.append(curr_indicator_column) # [(num_te,1), (num_te,1), ... ]
    pred_set_as_indicator_matrix = torch.cat(pred_set_as_indicator_matrix, dim=1) # (num_te, num_classes)
    # pred_set_as_indicator_matrix 0, 1 matrix for hard, 0~1 matrix for soft
    # using this, we can compute both inefficiency and validity for both evalulation and training
    return pred_set_as_indicator_matrix