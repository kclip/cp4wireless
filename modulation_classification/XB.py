import torch
import numpy as np
from general_utils import NC_compute, compute_prob_vec, compute_pred_set_as_indicator_matrix
import copy

class CrossVal_plus:
    def __init__(self, K, alpha, xi, num_classes, lr_inner=0.1, inner_iter=1, if_JK_mm=False, latch_best_tr=True):
        self.K =                K  # K-fold
        self.xi =               xi
        self.alpha =            alpha
        self.num_classes =      num_classes
        self.lr_inner =         lr_inner
        self.inner_iter =       inner_iter
        self.if_JK_mm =         if_JK_mm
        self.latch_best_tr =    latch_best_tr

    def forward(self, tr_dataset, X_te):
        # make tr_dataset into X and y
        # X_te: (num_te, *)
        # u_for_tr_dataset: (N, 1)
        # u_for_X_te: (num_te, 1)
        num_te =                        X_te.shape[0]
        N =                             tr_dataset[0].shape[0]  # len(tr_dataset)
        assert(                         N % self.K == 0)  # same-sized folds
        # split into K-folds
        m =                             N // self.K
        rand_perm_full_indices =        torch.randperm(N)
        dict_folds_indices =            {}
        dict_folds_curr_phis =          {}
        for ind_fold in range(self.K):
            dict_folds_indices[ind_fold] = rand_perm_full_indices[ind_fold * m:ind_fold * m + m]
            dict_folds_curr_phis[ind_fold] = None
        NC_y_dict =                     self.compute_NC_scores(
                                            tr_dataset,
                                            X_te,
                                            dict_folds_indices,
                                            dict_folds_curr_phis,
                                            self.latch_best_tr)
        pred_set_as_indicator_matrix =  compute_pred_set_as_indicator_matrix(
                                            NC_y_dict,
                                            N,
                                            self.alpha,
                                            self.num_classes,
                                            if_JK_mm=self.if_JK_mm)
        # pred_set_as_indicator_matrix 0, 1 matrix for hard, 0~1 matrix for soft
        # using this, we can compute both inefficiency and validity for both evalulation and training
        return pred_set_as_indicator_matrix

    def compute_NC_scores(self, tr_dataset, X_te, dict_folds_indices, dict_folds_curr_phis, latch_best_tr):
        xi =                            self.xi
        N =                             tr_dataset[0].shape[0]
        NC_y_dict =                     {}  # from i= 0, ..., N-1: {i: {y_grid: NC_ygrid}}
        # for each training point, fit model (phi) and get corresponding LOO NC score
        print(f'CV NC for {N} samples: ')
        for i in range(N):
            if i % 100 == 0:
                print(f'{i} ', end='')
            NC_y_dict[i] =              {}
            curr_x_tr =                 tr_dataset[0][i].unsqueeze(dim=0)  # this should be (1, *) , * for input size
            curr_y_tr =                 tr_dataset[1][i].unsqueeze(dim=0)  # this should be (1, 1) , single label
            fold_ind =                  assigning_fold(dict_folds_indices, i)
            if dict_folds_curr_phis[fold_ind] is not None:  # fitting for this fold already done
                with torch.no_grad():
                    curr_phi_LOO =      dict_folds_curr_phis[fold_ind]
            else:
                LOO_fold_dataset =      get_LOO_foldwise(
                                            tr_dataset,
                                            dict_folds_indices[fold_ind],
                                            N)
                curr_phi_LOO =          GD_for_XB(
                                            xi,
                                            LOO_fold_dataset,
                                            self.lr_inner,
                                            self.inner_iter,
                                            self.K-1,
                                            latch_best_tr)
                # save so that no need for refitting again for this fold
                with torch.no_grad():
                    dict_folds_curr_phis[fold_ind] = curr_phi_LOO
            with torch.no_grad():
                prob_vec_i_LOO =        compute_prob_vec(
                                            curr_phi_LOO,
                                            curr_x_tr,
                                            xi,
                                            self.num_classes)
                NC_i =                  NC_compute(
                                            prob_vec_i_LOO,
                                            torch.squeeze(curr_y_tr))
                NC_y_dict[i]['NC_i'] =  NC_i
                # for each training point, get corresponding NC score for new point (X_te: num_te, *)
                for y_prime in range(self.num_classes):
                    prob_vec_y_prime_LOO = compute_prob_vec(
                                            curr_phi_LOO,
                                            X_te,
                                            xi,
                                            self.num_classes)  # (num_te, num_classes)
                    NC_y_prime =        NC_compute(
                                            prob_vec_y_prime_LOO,
                                            y_prime)
                    NC_y_dict[i][y_prime] = NC_y_prime  # (num_te, 1)
        print('') # new line
        return NC_y_dict


def Trans(vector):
    return torch.transpose(vector, 0, 1)


def GD_for_XB_mb(xi, LOO_dataset, lr_inner, inner_iter,num_of_minibatches):
    print('GD_for_CV_mb training')
    phi =                   copy.deepcopy(xi)
    X_tr =                  LOO_dataset[0]
    Y_tr =                  LOO_dataset[1]
    size_of_minibatch =     X_tr.shape[0] // num_of_minibatches
    #print(f'CV GD_for_XB device of X_tr is {X_tr.device}')
    for iter in range(inner_iter):
        total_loss = 0.0
        for f in phi.parameters():
            f.total_grad = 0.0
        for iter_mb in range(num_of_minibatches):
            phi.zero_grad()
            out =               phi(X_tr[size_of_minibatch*iter_mb:size_of_minibatch*(iter_mb+1)], None)
            loss =              torch.nn.functional.cross_entropy(out, torch.squeeze(Y_tr)[size_of_minibatch*iter_mb:size_of_minibatch*(iter_mb+1)])
            total_loss +=       loss.item() / num_of_minibatches
            loss.backward()
            for f in phi.parameters():
                f.total_grad += f.grad.detach().clone().data
        for f in phi.parameters():
            f.data.sub_(        f.total_grad * lr_inner/num_of_minibatches)
        if iter%200==0:
            print(f'CV_mb tr iter {iter}/{inner_iter} loss= {(10*np.log10(total_loss))} dB')
    return list(map(lambda p: p[0], zip(phi.parameters())))


def GD_for_XB(xi, LOO_dataset, lr_inner, inner_iter,num_of_minibatches,latch_best_tr):
    print('GD_for_CV training')
    phi =                   copy.deepcopy(xi)
    X_tr =                  LOO_dataset[0]
    Y_tr =                  LOO_dataset[1]
    opt_nn =                copy.deepcopy(phi)
    min_loss =              torch.inf
    for iter in range(inner_iter):
        phi.zero_grad()
        out =               phi(X_tr, None)
        loss =              torch.nn.functional.cross_entropy(out, torch.squeeze(Y_tr))
        loss.backward()
        new_min_found =     (loss<min_loss)
        if latch_best_tr and new_min_found:
            opt_nn =        copy.deepcopy(phi)
            min_loss =      loss.detach().clone()
        for f in phi.parameters():
            f.data.sub_(f.grad.data * lr_inner)
        if iter%2==0:
            print(f'CV tr iter {iter}/{inner_iter} loss= {(10*np.log10(loss.item()))} dB {"***" if new_min_found else ""}')
    if latch_best_tr:
        return list(map(lambda p: p[0], zip(opt_nn.parameters())))
    else:
        return list(map(lambda p: p[0], zip(phi.parameters())))


def assigning_fold(dict_folds_indices, sample_ind):
    for ind_fold in dict_folds_indices.keys():
        if sample_ind in dict_folds_indices[ind_fold]:
            return ind_fold
        else:
            pass


def get_LOO_foldwise(tr_dataset, excluding_indices_list, N):
    including_indices_list =        [int(item) for item in torch.randperm(N) if item not in excluding_indices_list]
    LOO_X_tr =                      tr_dataset[0][including_indices_list]
    LOO_Y_tr =                      tr_dataset[1][including_indices_list]
    LOO_fold_dataset =              (LOO_X_tr, LOO_Y_tr)
    return LOO_fold_dataset
