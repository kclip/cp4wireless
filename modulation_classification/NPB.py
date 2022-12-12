import torch
import numpy as np
import copy

from general_utils import compute_pred_set_as_indicator_matrix
from torch.profiler import profile, record_function, ProfilerActivity


class Naive_Probabilistic_Based:
    def __init__(self, alpha, num_classes, c_sigmoid=None, c_sigmoid_for_ada_NC=None):
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, phi, tr_dataset, X_te):
        # here tr_dataset is used only for calibration
        # phi is already trained with training set
        m_prob_y_te =                       phi(X_te,None, if_softmax_out = True) # (N_te=X_te.shape[0],|Y|)
        m_sorted, m_ind =                   torch.sort(m_prob_y_te,dim=1,descending=True)
        m_sorted_cumsumed =                 torch.cumsum(m_sorted, dim =1)
        m_indictor_sorted =                 (m_sorted_cumsumed <= 1-self.alpha) # sorted indicator matrix of 0s and 1s
        v_last_sorted_member =              m_indictor_sorted.sum(dim=1).clip(max=self.num_classes - 1) # if all classes are within, no need to include
        m_indictor_sorted[torch.arange(m_indictor_sorted.shape[0]), v_last_sorted_member ] =   True # add the last member in each row (=test point) to complete 1-alpha
        pred_set_as_indicator_matrix =      m_indictor_sorted.gather(1, m_ind.argsort(1)) # reversing the sorted matrix back
        return pred_set_as_indicator_matrix

def GD_for_NPB_mb(xi, tr_set, lr_inner, inner_iter,num_of_minibatches): # due to GPU limited memory, GD is done by partitioning the training data set into num_of_minibatches mini-batches
    print('GD_for_NPB training')
    X_tr = tr_set[0]
    Y_tr = tr_set[1]
    size_of_minibatch = X_tr.shape[0] // num_of_minibatches
    optimizer = torch.optim.SGD(xi.parameters(), lr=lr_inner, momentum=0.5)
    for iter in range(inner_iter):
        total_loss =            0.0
        for f in xi.parameters():
            f.total_grad =      torch.zeros_like(f.data)
        for iter_mb in range(num_of_minibatches):
            xi.zero_grad()
            out =               xi(X_tr[size_of_minibatch*iter_mb:size_of_minibatch*(iter_mb+1)], None)
            loss =              torch.nn.functional.cross_entropy(out, torch.squeeze(Y_tr)[size_of_minibatch*iter_mb:size_of_minibatch*(iter_mb+1)])/num_of_minibatches
            total_loss +=       loss.item()
            loss.backward()
            nan_detected =          any([f.grad.data.isnan().any() for f in xi.parameters()]) # detects whether at least one entry of the gradient is nan
            if nan_detected:
                print(f'GD_for_NPB_mb training: nan detected, breaking the iteration loop at iter {iter}/{inner_iter} loss= {(10*np.log10(total_loss))} dB')
                break
            for f in xi.parameters():
                f.total_grad += torch.clamp(f.grad.detach().clone().data, min=-9999, max=9990)

        with torch.no_grad():
            for f in xi.parameters():
                f.grad.data =   f.total_grad
        optimizer.step()
        if iter%2==0:
            print(f'NPB_mb tr iter {iter}/{inner_iter} tr_loss= {(10*np.log10(total_loss))} dB')
        if nan_detected:
            break
    return xi

def Adam_for_NPB(xi, tr_set, lr_inner, inner_iter,num_of_minibatches): # due to GPU limited memory, GD is done by partitioning the training data set into num_of_minibatches mini-batches
    print('Adam_for_NPB training')
    X_tr = tr_set[0]
    Y_tr = tr_set[1]
    size_of_minibatch = X_tr.shape[0] // num_of_minibatches
    optimizer = torch.optim.Adam(xi.parameters(), lr=lr_inner)
    for iter in range(inner_iter):
        total_loss =            0.0
        for f in xi.parameters():
            f.total_grad =      0.0
        for iter_mb in range(num_of_minibatches):
            optimizer.zero_grad()
            out =               xi(X_tr[size_of_minibatch*iter_mb:size_of_minibatch*(iter_mb+1)], None)
            loss =              torch.nn.functional.cross_entropy(out, torch.squeeze(Y_tr)[size_of_minibatch*iter_mb:size_of_minibatch*(iter_mb+1)])/num_of_minibatches
            total_loss +=       loss.item()
            loss.backward()
            nan_detected =          any([f.grad.data.isnan().any() for f in xi.parameters()]) # detects whether at least one entry of the gradient is nan
            if nan_detected:
                print(f'Adam_for_NPB training: nan detected, breaking the iteration loop at iter {iter}/{inner_iter} loss= {(10*np.log10(total_loss))} dB')
                for f in xi.parameters():
                    print(f.grad.detach().clone().data)
                    print(f.total_grad)
                break
            for f in xi.parameters():
                f.total_grad += f.grad.detach().clone().data
        with torch.no_grad():
            for f in xi.parameters():
                f.grad.data =   f.total_grad
        optimizer.step()
        if iter%1==0:
            print(f'NPB tr iter {iter}/{inner_iter} loss= {(10*np.log10(total_loss))} dB')
        if nan_detected:
            break
    return xi

def GD_for_NPB(xi, tr_set, lr_inner, inner_iter,num_of_minibatches,latch_best_tr):
    print('GD_for_NPB training')
    X_tr =                  tr_set[0]
    Y_tr =                  tr_set[1]
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
            f.data.sub_(    f.grad.data * lr_inner)
        if iter%2==0:
            print(f'NPB tr iter {iter}/{inner_iter} loss= {(10*torch.log10(loss)).item()} dB {"***" if new_min_found else ""}')
    return opt_nn if latch_best_tr else xi




