import torch
import numpy as np
import copy
from general_utils import NC_compute, compute_prob_vec, compute_pred_set_as_indicator_matrix

class Split_Conformal:
    def __init__(self, alpha, num_classes, c_sigmoid=None, c_sigmoid_for_ada_NC=None):
        self.alpha = alpha
        self.num_classes = num_classes
    def forward(self, phi, tr_dataset, X_te):
        # here tr_dataset is used only for calibration
        # phi is already trained with proper training set 
        num_te = X_te.shape[0]

        N = tr_dataset[0].shape[0]
        NC_y_dict = self.compute_NC_scores(phi, tr_dataset, X_te)
        pred_set_as_indicator_matrix = compute_pred_set_as_indicator_matrix(NC_y_dict, N, self.alpha, self.num_classes, if_JK_mm=False)
        return pred_set_as_indicator_matrix
    
    def compute_NC_scores(self, curr_phi, tr_dataset, X_te):
        #  tr_dataset is used for validation set
        # fitting is already done -- curr_phi
        N = tr_dataset[0].shape[0] #len(tr_dataset)
        NC_y_dict = {} # from i= 0, ..., N-1: {i: {y_grid: NC_ygrid}}
        # for each training point, fit model (phi) and get corresponding LOO NC score
        print(f'VB NC for {N} samples: ')
        with torch.no_grad():
            curr_phi_para_list = list(map(lambda p: p[0], zip(curr_phi.parameters())))
            for i in range(N):
                if i%100==0:
                    print(f'{i} ', end='')
                NC_y_dict[i] = {}
                curr_x_tr = tr_dataset[0][i].unsqueeze(dim=0)  # this should be (1, *) , * for input size
                curr_y_tr = tr_dataset[1][i].unsqueeze(dim=0)  # this should be (1, 1) , single label
                prob_vec_i = compute_prob_vec(curr_phi_para_list, curr_x_tr, curr_phi, self.num_classes)
                NC_i = NC_compute(prob_vec_i, torch.squeeze(curr_y_tr))
                NC_y_dict[i]['NC_i'] = NC_i # (1, 1)
                # for each training point, get corresponding NC score for new point (X_te: num_te, *)
                for y_prime in range(self.num_classes):
                    prob_vec_y_prime = compute_prob_vec(curr_phi_para_list, X_te, curr_phi, self.num_classes) # (num_te, num_classes)
                    NC_y_prime = NC_compute(prob_vec_y_prime, y_prime)
                    NC_y_dict[i][y_prime] = NC_y_prime # (num_te, 1)
                    # this is same for every i but for the consistency with code with CV, we keep it as this.
            print('')
        return NC_y_dict


def GD_for_VB_mb(xi, proper_tr_set, lr_inner, inner_iter,num_of_minibatches): # due to GPU limited memory, GD is done by partitioning the training data set into num_of_minibatches mini-batches
    print('GD_for_VB training')
    X_tr =                  proper_tr_set[0]
    Y_tr =                  proper_tr_set[1]
    size_of_minibatch =     X_tr.shape[0] // num_of_minibatches
    for iter in range(inner_iter):
        total_loss =        0.0
        for f in xi.parameters():
            f.total_grad =  0.0
        for iter_mb in range(num_of_minibatches):
            xi.zero_grad()
            out =           xi(X_tr[size_of_minibatch*iter_mb:size_of_minibatch*(iter_mb+1)], None)
            loss =          torch.nn.functional.cross_entropy(out, torch.squeeze(Y_tr[size_of_minibatch*iter_mb:size_of_minibatch*(iter_mb+1)]))
            total_loss +=   loss.item() / num_of_minibatches
            loss.backward()
            for f in xi.parameters():
                f.total_grad += f.grad.detach().clone().data
        for f in xi.parameters():
            f.data.sub_(f.total_grad * lr_inner/num_of_minibatches)
        if iter%200==0:
            print(f'VB_mb tr iter {iter}/{inner_iter} loss= {(10*np.log10(total_loss)).item()} dB')
    return xi

def GD_for_VB(xi, proper_tr_set, lr_inner, inner_iter,num_of_minibatches,latch_best_tr): # due to GPU limited memory, GD is done by partitioning the training data set into num_of_minibatches mini-batches
    print('GD_for_VB training')
    X_tr =                  proper_tr_set[0]
    Y_tr =                  proper_tr_set[1]
    opt_nn =                copy.deepcopy(xi)
    min_loss =              torch.inf
    for iter in range(inner_iter):
        xi.zero_grad()
        out =               xi(X_tr, None)
        loss =              torch.nn.functional.cross_entropy(out, torch.squeeze(Y_tr))
        loss.backward()
        new_min_found =     (loss<min_loss)
        if latch_best_tr and new_min_found:
            opt_nn =        copy.deepcopy(xi)
            min_loss =      loss.detach().clone()
        for f in xi.parameters():
            f.data.sub_(f.grad.data * lr_inner)
        if iter%2==0:
            print(f'VB tr iter {iter}/{inner_iter} loss= {(10*np.log10(loss.item()))} dB {"***" if new_min_found else ""}')
    return opt_nn if latch_best_tr else xi