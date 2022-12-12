
# algebra and matrices package
import numpy as np

import matplotlib.pyplot as plt
#import seaborn as sns
#import pandas as pd
from scipy.io import loadmat, savemat

# Machine learning packages
from torch import nn
import torch
import torch.optim as optim

import datetime
from copy import deepcopy


COMPUTE_BAYES =                     True
REPARAM_COEFF =                     0.1
FORCE_AWGN =                        True

def xconjx(x): # squared l2 norm for complex ndarrays
    return x.real*x.real + x.imag*x.imag

class Modulation:
    def __init__(self, modKey):
        self.modKey = modKey
        dOrders = {  'WGN': [],
                    'QPSK':  4,
                   '8APSK':  8,
                   '16QAM': 16,
                   '64QAM': 64,
                    '8GAM':  8,
                   '16GAM': 16}
        dNormCoeffs = {  'WGN': [],
                        'QPSK': torch.tensor(1 /  2.0).sqrt(),
                       '8APSK': torch.tensor(1.0), # the two rings have unit power
                       '16QAM': torch.tensor(1 / 10.0).sqrt(),
                       '64QAM': torch.tensor(1 / 42.0).sqrt(),
                        '8GAM': (1/torch.arange(1.0, 8+1)).sum().sqrt(),
                       '16GAM': (1/torch.arange(1.0,16+1)).sum().sqrt()}
        # mean unit power for the 8APSK:
        apsk8_inner_radius =        2.0/torch.sqrt(2.0+(1+torch.sqrt(torch.tensor(3.0)))**2)    # ~= 0.65
        apsk8_outer_radius =        (2.0 - apsk8_inner_radius**2).sqrt()                        # ~= 1.26.
        
        dMappings = {  'WGN': [],
                      'QPSK': torch.tensor( [+1.+1.j, -1.+1.j, +1.-1.j, -1.-1.j  ],
                                            dtype = torch.complex64),
                      '8APSK': torch.cat(
                                [torch.tensor([+1.+1.j, -1.+1.j, +1.-1.j, -1.-1.j], dtype=torch.complex64) * apsk8_inner_radius/torch.tensor(2.0).sqrt(),
                                 torch.tensor([+1.,     +1.j,    -1,      -1.j   ], dtype=torch.complex64) * apsk8_outer_radius
                                ]),
                    # conventional 16QAM
                    #'16QAM': torch.tensor( [+1.+1.j, +1.+3.j, +3.+1.j, +3.+3.j, +1.-1.j, +1.-3.j, +3.-1.j, +3.-3.j,
                    #                        -1.+1.j, -1.+3.j, -3.+1.j, -3.+3.j, -1.-1.j, -1.-3.j, -3.-1.j, -3.-3.j  ]),
                    # Sangwoo's 16QAM
                     '16QAM': torch.tensor( [-3.-3.j, -3.+1.j, +1.+1.j, +1.-3.j, -3.+3.j, +3.+1.j, +1.-1.j, -1.-3.j,
                                             +3.+3.j, +3.-1.j, -1.-1.j, -1.+3.j, +3.-3.j, -3.-1.j, -1.+1.j, +1.+3.j, ],
                                            dtype = torch.complex64),
                     '64QAM': torch.tensor( [-7.-7.j, -5.-7.j, -3.-7.j, -1.-7.j, +1.-7.j, +3.-7.j, +5.-7.j, +7.-7.j,
                                             -7.-5.j, -5.-5.j, -3.-5.j, -1.-5.j, +1.-5.j, +3.-5.j, +5.-5.j, +7.-5.j,
                                             -7.-3.j, -5.-3.j, -3.-3.j, -1.-3.j, +1.-3.j, +3.-3.j, +5.-3.j, +7.-3.j,
                                             -7.-1.j, -5.-1.j, -3.-1.j, -1.-1.j, +1.-1.j, +3.-1.j, +5.-1.j, +7.-1.j,
                                             -7.+1.j, -5.+1.j, -3.+1.j, -1.+1.j, +1.+1.j, +3.+1.j, +5.+1.j, +7.+1.j,
                                             -7.+3.j, -5.+3.j, -3.+3.j, -1.+3.j, +1.+3.j, +3.+3.j, +5.+3.j, +7.+3.j,
                                             -7.+5.j, -5.+5.j, -3.+5.j, -1.+5.j, +1.+5.j, +3.+5.j, +5.+5.j, +7.+5.j,
                                             -7.+7.j, -5.+7.j, -3.+7.j, -1.+7.j, +1.+7.j, +3.+7.j, +5.+7.j, +7.+7.j  ],
                                            dtype = torch.complex64),
                      '8GAM': 1/torch.arange(1.0, 8+1) * torch.exp(1.j*2*np.pi*((3-np.sqrt(5))/2) * torch.arange(1.0, 8+1)),
                     '16GAM': 1/torch.arange(1.0,16+1) * torch.exp(1.j*2*np.pi*((3-np.sqrt(5))/2) * torch.arange(1.0,16+1))}
        # temporarily 64QAM (not gray, simple meshgrid)! complete later 64QAM using
        # http://literature.cdn.keysight.com/litweb/pdf/ads2008/3gpplte/ads2008/LTE_Mapper_(Mapper).html
        self.order = dOrders[modKey]
        self.normCoeff = dNormCoeffs[modKey]
        self.vMapping = dMappings[modKey]

    def modulate(self,vSamplesUint):
        return self.normCoeff * self.vMapping[vSamplesUint]

    def step(self, numSamples , xPattern):
        if self.modKey == 'WGN':
            vSamplesUint = torch.zeros(numSamples)
            vSamplesIq = (       torch.randn(numSamples)
                          + 1j * torch.randn(numSamples)    ) / torch.sqrt(2.0)
        else:
            if xPattern==0:             # 0,1,2,3,0,1,2,3,0,...
                vSamplesUint =          (torch.arange(0,numSamples, dtype = torch.long)   %self.order)
            elif xPattern==1:           # 0,0,0,0,1,1,1,1,2,...
                vSamplesUint =          (torch.arange(0, numSamples, dtype = torch.long)//self.order)%self.order
            else:  # some random balanced permutation of (0,1,...M-1,0,1,...M-1,...)
                vSamplesUint =          torch.randperm(numSamples) % self.order
            vSamplesIq =                self.modulate(vSamplesUint)
        return (vSamplesIq, vSamplesUint)

    def hard_demodulator(self, vRx_iq):
        vHardRxUint = torch.argmin(torch.abs(   ( (vRx_iq.real+1j*vRx_iq.imag)
                                                - self.normCoeff * self.vMapping.unsqueeze(0)  ) ) ,
                                    axis=0)
        return vHardRxUint

    def plot_decision_borders(self):
        if self.modKey == 'QPSK':
            plt.plot([-2, 2], [0, 0], 'k--')
            plt.plot([0, 0], [-2, 2], 'k--')
        elif self.modKey == '16QAM' or self.modKey == '64QAM':
            dBordersOneDim = {'16QAM': np.arange(-2, 2 + 1, 2) * 1 / np.sqrt(10),
                              '64QAM': np.arange(-6, 6 + 1, 2) * 1 / np.sqrt(42)}
            vBorders = dBordersOneDim[self.modKey]
            delta = vBorders[2] - vBorders[1]
            for i in vBorders:
                plt.plot([i, i], [vBorders[0] - delta, vBorders[-1] + delta], 'k--')
                plt.plot([vBorders[0] - delta, vBorders[-1] + delta], [i, i], 'k--')

class IotUplink:

    def __init__(self, dSetting):
        self.dSetting =                 dSetting
        self.modulator =                Modulation(dSetting['modKey'])
        self.snr_lin =                  10.0 ** (self.dSetting['snr_dB'] / 10)
        self.randbeta =                 torch.distributions.beta.Beta(5, 2)

    def draw_channel_state(self): # samples new state to be used later
        ch_mult_factor =                    torch.randn(1, dtype = torch.cfloat) # no need for torch.tensor(0.5).sqrt()
        if FORCE_AWGN:
            ch_mult_factor /=               ch_mult_factor.abs() # instead of Rayleigh fading, make it AWGN with fixed SNR and arbitrary complex phase
        self.dChannelState = {
            'tx_amp_imbalance_factor':      self.randbeta.sample()*0.15, # in [0,0.15]
            'tx_phase_imbalance_factor':    self.randbeta.sample()*15*3.14159265/180 ,#in [0,15] deg
            'ch_mult_factor':               ch_mult_factor
            }

    def step(self, numSamples , bEnforcePattern, bNoiseFree = False):
        if bEnforcePattern:
            pattern =       0
        else:
            pattern =       -1
        txIq , txSym =      self.modulator.step(numSamples, pattern)
        epsilon =           self.dChannelState['tx_amp_imbalance_factor']
        cos_delta =         torch.cos(self.dChannelState['tx_phase_imbalance_factor'])
        sin_delta =         torch.sin(self.dChannelState['tx_phase_imbalance_factor'])
        txDistorted =       (      (1+epsilon)*(cos_delta * txIq.real - sin_delta * txIq.imag)
                             +1j * (1-epsilon)*(cos_delta * txIq.imag - sin_delta * txIq.real) ) # TX non-lin
        txRayleighed =      txDistorted * self.dChannelState['ch_mult_factor'] # complex mult
        if bNoiseFree:
            noise =         0
        else:
            noise  =        np.sqrt(0.5 /self.snr_lin)*  torch.randn(numSamples, dtype=torch.cfloat)
        rxIq =              txRayleighed + noise
        rxReal =            torch.tensor([[z.real,z.imag] for z in rxIq], dtype = torch.float64) # complex to two-dim tensor
        return MyDataSet(   {       'X': rxReal,
                                    'y': txSym  } )
class MyDataSet(torch.utils.data.Dataset):

    # Constructor
    def __init__ ( self, dDataset):   # dDataset['y'] are the inputs and dDataset['x'] (long) are the outputs on purpose
        self.X =        dDataset['X']
        self.y =        dDataset['y']
        self.len =      len(self.y)

    # Getter
    def __getitem__ ( self, index ):
        return self.X[index], self.y[index]

    # Get length
    def __len__ ( self ):
        return self.len

    def leave_one_out(self, index):
        indices_loo =           np.concatenate((np.arange(0,index),np.arange(index+1,len(self))),axis=0) # loo = leave one out
        return MyDataSet({'X': self.X[indices_loo, : ],
                          'y': self.y[indices_loo    ]})

    def leave_fold_out(self, index, numfolds):
        NoverK =                   round(len(self) / numfolds)
        indices_lfo =           np.concatenate((np.arange(0,index*NoverK),np.arange((index+1)*NoverK,len(self))),axis=0) # loo = leave one out
        return MyDataSet({'X': self.X[indices_lfo, : ],
                          'y': self.y[indices_lfo    ]})


    def split_into_two_subsets( self, numSamFirstSubSet, bShuffle = True ): # including shuffling
        N =             self.len
        N0 =            numSamFirstSubSet
        N1 =            N - N0
        if N0==N:
            return (    MyDataSet({     'X': self.X,
                                        'y': self.y  } ),
                        MyDataSet({     'X': [],
                                        'y': []  } )       )
        elif N1==N:
            return (    MyDataSet({     'X': [],
                                        'y': []}),
                        MyDataSet({     'X': self.X,
                                        'y': self.y }  )   )
        else:
            if bShuffle:
                perm =  torch.randperm(N)
            else:
                perm =  torch.arange(0,N)
            return (    MyDataSet({     'X': self.X[perm[    : N0],:],
                                        'y': self.y[perm[    : N0]  ]  } ),
                        MyDataSet({     'X': self.X[perm[-N1 :   ],:],
                                        'y': self.y[perm[-N1 :   ]  ]  } ) )

class FcReluDnn(nn.Module):     # Fully-Connected ReLU Deep Neural Network
    # Constructor
    def __init__ ( self, vLayers ):
        super(FcReluDnn, self).__init__()
        self.hidden = nn.ModuleList()
        for input_size, output_size in zip(vLayers, vLayers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size, dtype = torch.float64) ) # for Hessian calc, it is better to go from single to double

    # Prediction
    def forward ( self, activation ):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                activation = torch.nn.functional.relu(linear_transform(activation))
            else:
                activation = linear_transform(activation)
        return activation

    def num_parameters( self ):
        return sum([torch.numel(w) for w in self.parameters()])


class FcReluDnn_external(nn.Module):     # Fully-Connected ReLU Deep Neural Network using external model parameters
    # Constructor
    def __init__ ( self ):
        super(FcReluDnn_external, self).__init__() # no need to initialize nn since we are given with parameter list
    # Prediction
    def forward ( self, net_in , net_params): # net_in is the input and output. net_params is the nn parameter vector
        L = int(len(net_params)/2) # 2 for weight+bias in the parameter list; L stands for number of layers
        for ll in range(L):
            curr_layer_weight = net_params[2*ll  ]
            curr_layer_bias =   net_params[2*ll+1]
            net_in =            torch.nn.functional.linear(net_in, curr_layer_weight, curr_layer_bias)
            if ll<L-1: # inner layer has ReLU activation, last layer doesn't. Its softmax is in the loss func
                net_in=         torch.nn.functional.relu(net_in)
        return net_in


def fitting_erm_ml__gd(             model,      # will be changed while trained
                                    D,
                                    D_te,       # only for evaluating the loss_te. model doesn't take it into account
                                    gd_init_sd, # should be of the same parameter dim of the model above
                                    gd_lr,
                                    gd_num_iters
                                ):
    model.load_state_dict(      gd_init_sd)
    criterion =                 nn.CrossEntropyLoss()
    # we don't the optimizer directly, as to resemble operation in the MAP counter function, they are given here for clarity
    #optimizer =                 optim.Adam(             model.parameters(), lr =            gd_lr)
    v_loss_tr =                 torch.zeros(gd_num_iters, dtype=torch.float)
    v_loss_te =                 torch.zeros(gd_num_iters, dtype=torch.float)
    for i_iter in range(gd_num_iters):
        v_loss_te[i_iter] =     criterion(model(D_te.X), D_te.y).detach()
        for w in model.parameters(): # we loop, instead of "optimizer.zero_grad()"
            w.grad =            torch.zeros_like(w.data)
        loss =                  criterion(model(D.X), D.y) # GD and not SGD
        loss.backward()
        for w in model.parameters(): # we loop, instead of "optimizer.step()"
            w.data -=           gd_lr * w.grad.data
        v_loss_tr[i_iter] =     loss.detach()
    return v_loss_tr, v_loss_te


def fitting_erm_map_gd(             model, # will be changed while trained
                                    D,
                                    D_te,       # only for evaluating the loss_te. model doesn't take it into account
                                    gd_init_sd, # should be of the same parameter dim of the model above
                                    gd_lr,
                                    gd_num_iters,
                                    gamma,           # 0 = ML ; >0 = MAP
                                    ensemble_size,
                                    compute_hessian, #{0: all zero hessian = faster = nonsense, 1: model-based using external network, 2: grad(grad) model agnoistic}
                                    lmc_burn_in,
                                    lmc_lr_init,
                                    lmc_lr_decaying
                                ):
    model.load_state_dict(      gd_init_sd)
    lmc_model =                 deepcopy(model) # we assume lmc_init_sd = gd_init_sd
    ext_model =                 FcReluDnn_external()
    criterion =                 nn.CrossEntropyLoss() # for MAP, we will L2 regularize using the gradient update, not through the loss
    N =                         len(D)
    v_loss_tr =                     torch.zeros(gd_num_iters, dtype=torch.float64)
    v_loss_te =                     torch.zeros(gd_num_iters, dtype=torch.float64)
    # training the MAP solution
    for i_iter in range(gd_num_iters):
        v_loss_te[i_iter] =         criterion(model(D_te.X), D_te.y).detach() # evaluating the test loss without the L2
        for w in model.parameters(): # we loop, instead of "optimizer.zero_grad()"
            w.grad =                torch.zeros_like(w.data)
        loss =                      criterion(model(D.X), D.y) # GD and not SGD, over the training set  # likelihood
        loss.backward()
        for w in model.parameters(): # we loop, instead of "optimizer.step()"
            w.data +=               - gd_lr * (w.grad.data + gamma/N * w.data ) # the second term is the grad of the l2 regularization
        v_loss_tr[i_iter] =         loss.detach()
    v_phi_map =                     torch.nn.utils.parameters_to_vector(model.parameters()).double()
    ensemble_vectors_lmc =          torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone().repeat(ensemble_size,1)
    ensemble_vectors_hes =          deepcopy(ensemble_vectors_lmc)  # will stay such if compute_hessian==0. If 1 or 2, will be overwritten
    if COMPUTE_BAYES:
        # 1) LMC approach = Langevin Monte Carlo (= SGLD) for creating an ensemble
        lmc_num_iters =             lmc_burn_in + ensemble_size - 1
        if lmc_lr_decaying > 1.0:
            lmc_lr_last =           lmc_lr_init / lmc_lr_decaying
            # lmc_lr will decrease exponentially form lmc_lr to lmc_lr_last using a*((b+t)**(-gamma)) for time index t=0,1,2,...
            lmc_lr_gamma =          0.55
            lmc_lr_b =              (lmc_num_iters - 1) / ( ((lmc_lr_init/lmc_lr_last)**(1/lmc_lr_gamma)) - 1)
            lmc_lr_a =              lmc_lr_init * (lmc_lr_b**lmc_lr_gamma)
        else: # constant lr
            lmc_lr_gamma =          0
            lmc_lr_b =              0
            lmc_lr_a =              lmc_lr_init
        v_lmc_loss_tr =             torch.zeros(lmc_num_iters, dtype=torch.float)
        v_lmc_loss_te =             torch.zeros(lmc_num_iters, dtype=torch.float)
        lmc_temperature =           20.0  # Langevin MC temperature
        for i_iter in range(lmc_num_iters):
            lmc_lr =                    lmc_lr_a * (lmc_lr_b + i_iter)**(-lmc_lr_gamma)
            v_lmc_loss_te[i_iter] =     criterion(lmc_model(D_te.X), D_te.y).detach()
            for w in lmc_model.parameters():
                w.grad =                torch.zeros_like(w.data)
            loss =                      criterion(lmc_model(D.X), D.y) # GD and not SGD, over the training set  # likelihood
            loss.backward()
            for w in lmc_model.parameters(): # we loop, instead of "optimizer.step()"
                if 0:#(i_iter + 1 < lmc_burn_in):  # while in burn-in, don't add noise
                    w.data +=               - lmc_lr * (w.grad.data + gamma/N * w.data )
                else: # after burn-in
                    w.data +=               - lmc_lr * (w.grad.data + gamma/N * w.data ) + np.sqrt(2*lmc_lr / (N*lmc_temperature)) * torch.randn_like(w.data) # with noise injected
            if (i_iter + 1 >= lmc_burn_in): # after the burn-in, we save each model parameter vector to form the ensemble
                r =                 i_iter + 1 - lmc_burn_in
                ensemble_vectors_lmc[r,:] =         torch.nn.utils.parameters_to_vector( lmc_model.parameters() )
            v_lmc_loss_tr[i_iter] = loss.detach()
        # 2) the Hessian approaches for Sigma_MAP for creating an ensemble
        if compute_hessian==1:
            model_for_hessian =         deepcopy(model)
            for w in model_for_hessian.parameters(): # we loop, instead of "optimizer.zero_grad()"
                w.grad =                torch.zeros_like(w.data)
            T =                         len(tuple(model_for_hessian.parameters()))
            def loss_for_hessian(*in_params):
                for ii in range(len(model_for_hessian.hidden)):
                    del                         model_for_hessian.hidden[ii].weight
                    model_for_hessian.hidden[ii].weight =   in_params[2*ii+0]
                    del                         model_for_hessian.hidden[ii].bias
                    model_for_hessian.hidden[ii].bias =     in_params[2*ii+1]
                return criterion(ext_model(D.X,in_params), D.y)
            t_hessian =                 torch.autograd.functional.hessian(  loss_for_hessian,
                                                                            tuple(model_for_hessian.parameters()))#,vectorize=True)
            m_hessian = torch.tensor([])
            for row in range(T):
                # the dim keeps changing, due to singleton dims removed. This helps to reconstruct them:
                if row%2==0:                    # indices for row: [0, 1], indices for col either [2] or [2,3]
                    dim0_len =          t_hessian[row][0].shape[0] * t_hessian[row][0].shape[1]
                else:                           # indices for row: [0], indices for col either [1,2] or [1]
                    dim0_len =          t_hessian[row][0].shape[0]
                blocked_row =           torch.cat([t_hessian[row][col].view(dim0_len,-1) for col in range(T)], dim=1)
                m_hessian =             torch.cat([m_hessian, blocked_row], dim=0)
            m_hessian =                 0.5*(m_hessian + m_hessian.T) # remove asymmetry caused by numerical instability (order of diff)
        elif compute_hessian==2:
            model_for_hessian =         deepcopy(model)
            for w in model_for_hessian.parameters(): # we loop, instead of "optimizer.zero_grad()"
                w.grad =                torch.zeros_like(w.data)
            loss_for_hessian_grad2 =    criterion(model_for_hessian(D.X), D.y)
            loss_grad =                 torch.autograd.grad(loss_for_hessian_grad2,
                                                            model_for_hessian.parameters(),
                                                            create_graph=True)
            m_hessian =                 eval_hessian(loss_grad, model_for_hessian)
            m_hessian =                 0.5*(m_hessian + m_hessian.T) # remove asymmetry caused by numerical instability (order of diff)
        # 3) the empirical FIM approach for Sigma_MAP for creating an ensemble
        m_fim_D =                   torch.zeros((len(v_phi_map),len(v_phi_map)), dtype=torch.float64) # matrix, empirical Fisher Information Matrix
        for x,y in D:
            for w in model.parameters(): # we loop, instead of "optimizer.zero_grad()"
                w.grad.data =       torch.zeros_like(w.data)
            loss =                  criterion(model(x.view(1,-1)), y.view(-1)) # training loss of one sample
            v_grad =                torch.nn.utils.parameters_to_vector(    torch.autograd.grad(loss,model.parameters()) )
            m_fim_D +=              torch.mm( v_grad.view(-1, 1),
                                              v_grad.view( 1,-1)  )
        m_fim_D /=                  N # normalization
        # *) Hessian and its FIM approx -> Ensembles
        if compute_hessian==0:
            m_precision_map_hes =   torch.diag(torch.full([len(v_phi_map)],np.inf,dtype=torch.float64))          #matrix, inf precision
        else:
            m_precision_map_hes =   (gamma * torch.eye(len(v_phi_map)) + N * m_hessian)   # matrix, no inversion for precision
        m_precision_map_fim =       (gamma * torch.eye(len(v_phi_map)) + N * m_fim_D  )   # matrix, no inversion for precision
        mvn_hes =                   torch.distributions.multivariate_normal.MultivariateNormal(
                                                                                loc =               v_phi_map,           # mean vec
                                                                                precision_matrix =  m_precision_map_hes) # precision matrix
        mvn_fim =                   torch.distributions.multivariate_normal.MultivariateNormal(
                                                                                loc =               v_phi_map,           # mean vec
                                                                                precision_matrix =  m_precision_map_fim) # precision matrix
        ensemble_vectors_hes =      mvn_hes.sample(sample_shape=[ensemble_size]) # size = [ensemble_R,dim_phi]
        ensemble_vectors_fim =      mvn_fim.sample(sample_shape=[ensemble_size])
    return v_loss_tr, v_loss_te, v_lmc_loss_tr, v_lmc_loss_te, ensemble_vectors_hes, ensemble_vectors_fim, ensemble_vectors_lmc


# eval Hessian matrix
def eval_hessian(loss_grad, model):
    # courtesy of paul_C in https://discuss.pytorch.org/t/compute-the-hessian-matrix-of-a-network/15270/3
    # note: loss_grad is defined before calling as torch.autograd.grad(loss, net.parameters(), create_graph=True)
    cnt = 0
    for g in loss_grad:
        g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
        cnt = 1
    l = g_vector.size(0)
    hessian = torch.zeros((l, l),dtype=torch.float64)
    for idx in range(l):
        grad2rd = torch.autograd.grad(g_vector[idx], model.parameters(), create_graph=True)
        cnt = 0
        for g in grad2rd:
            g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
            cnt = 1
        hessian[idx] = g2
    return hessian


def ensemble_predict(   X, ensemble_vectors, model):
    # loop over the ensemble, and take the mean of the predictions
    R =                             ensemble_vectors.size(dim=0)
    for r in range(R):
        torch.nn.utils.vector_to_parameters(        ensemble_vectors[r,:],
                                                    model.parameters())
        if r==0:
            m_prob_y =              torch.nn.functional.softmax(model(X), dim=1)
        else:
            m_prob_y +=             torch.nn.functional.softmax(model(X), dim=1)
    m_prob_y /=                     R
    return m_prob_y

def nonconformity_frq(X, y, model):
    m_prob_y =                          torch.nn.functional.softmax(model(X), dim=1)
    return -torch.log(m_prob_y[torch.arange(len(y)),y])

def nonconformity_bay(X, y, model, ensemble_vectors):
    ens_pred =                          ensemble_predict(X, ensemble_vectors, model)
    return -torch.log(ens_pred[torch.arange(len(y)),y])

def nonconformity_frq_giq(X, y, model): # giq = generalized inverse quantile
    m_prob_y_te =                       torch.nn.functional.softmax(model(X), dim=1)
    m_sorted, m_ind =                   torch.sort(m_prob_y_te,dim=1,descending=True)
    v_NC =                              torch.zeros(len(y), dtype=torch.float )
    for i_te in range(len(y)):
        i_y =                           0
        while True:
            v_NC[i_te] +=               m_sorted[i_te,i_y]
            if (m_ind[i_te,i_y]==y[i_te]):
                break
            i_y +=                      1
    return v_NC

def nonconformity_bay_giq(X, y, model, ensemble_vectors): # giq = generalized inverse quantile
    m_prob_y_te =                       ensemble_predict(X, ensemble_vectors, model)
    m_sorted, m_ind =                   torch.sort(m_prob_y_te,dim=1,descending=True)
    v_NC =                              torch.zeros(len(y), dtype=torch.float )
    for i_te in range(len(y)):
        i_y =                           0
        while True:
            v_NC[i_te] +=               m_sorted[i_te,i_y]
            if (m_ind[i_te,i_y]==y[i_te]):
                break
            i_y +=                      1
    return v_NC

def quantile_from_top(vec, alpha):
    torch_inf =         torch.tensor(torch.inf).unsqueeze(0)
    sorted, _ =         torch.sort(torch.cat((vec, torch_inf)))
    return              sorted[int(np.ceil( (1-alpha)*(len(vec)+1)-1 ))]

def quantile_from_btm(vec, alpha):
    return              -quantile_from_top(-vec, alpha)

def sft_covrg_and_ineff(m_prob_y_te,y_te,alpha):
    m_sorted, m_ind =                   torch.sort(m_prob_y_te,dim=1,descending=True)
    n_y =                               m_prob_y_te.shape[1]
    l_covrg_labels =                    [ [] for _ in range(n_y) ] # list of lists
    l_ineff_labels =                    [ [] for _ in range(n_y) ] # list of lists
    v_set_size =                        torch.ones( len(y_te), dtype=torch.long )
    v_is_in_set =                       torch.zeros(len(y_te), dtype=torch.bool )
    for i_te in range(len(y_te)):
        v_is_in_set[i_te] =             m_ind[i_te,0]==y_te[i_te] # the first label (with highest soft prob) is surly in the predictive set
        cum_prob =                      m_sorted[i_te,0]  # and so does its prob
        while (cum_prob<1-alpha) and (v_set_size[i_te]<n_y):
            v_set_size[i_te] +=         1
            cum_prob +=                 m_sorted[i_te,v_set_size[i_te]-1]
            v_is_in_set[i_te] |=        m_ind[i_te,v_set_size[i_te]-1]==y_te[i_te]
        l_covrg_labels[y_te[i_te]].append( v_is_in_set[i_te].item())
        l_ineff_labels[y_te[i_te]].append( v_set_size[i_te].item())
    ineff =                             v_set_size.float().mean()
    covrg =                             v_is_in_set.float().mean()
    v_covrg_labels =                    [torch.tensor(l).float().mean() for l in l_covrg_labels] # list of lists to vector
    v_ineff_labels =                    [torch.tensor(l).float().mean() for l in l_ineff_labels] # list of lists to vector
    return covrg, ineff, v_covrg_labels, v_ineff_labels

def vb__covrg_and_ineff(m_NC_prs, v_NC_val, y_te, alpha): # prs = prospective
    quan_val =                      quantile_from_top(v_NC_val, alpha) # scalar
    m_is_prs_in_pred_set =          (m_NC_prs <= quan_val).float()       # Boolean matrix, [N_te,Y_prime]
    v_is_y_te_in_pred =             m_is_prs_in_pred_set[torch.arange(len(y_te)),y_te]
    covrg =                         v_is_y_te_in_pred.mean()    # mean of vector
    n_y =                           m_is_prs_in_pred_set.shape[1]
    v_covrg_labels =                torch.zeros(n_y, dtype=torch.float)
    v_ineff_labels =                torch.zeros(n_y, dtype=torch.float)
    for i_y in torch.arange(n_y):
        indices_i_y =               (y_te == i_y).nonzero().view(-1)
        v_covrg_labels[i_y] =       m_is_prs_in_pred_set[indices_i_y, y_te[indices_i_y]].mean()
        v_ineff_labels[i_y] =       m_is_prs_in_pred_set[indices_i_y,:].sum(dim=1).mean()
    ineff =                         m_is_prs_in_pred_set.sum(dim=1).mean() # for each i_te, sum all y's. Then mean over the test
    return covrg, ineff, v_covrg_labels, v_ineff_labels

def jkp_covrg_and_ineff(m_NC_prs, v_NC, y_te, alpha): # prs = prospective   ; size of matrix [N_te,N_prime,N]; of vector [N]
    th =                            (1-alpha)*(len(v_NC)+1)
    m_is_prs_in_pred_set =          ((m_NC_prs > v_NC.view(1,1,-1)).float().sum(dim=2) < th).float()
    v_is_y_te_in_pred =             m_is_prs_in_pred_set[torch.arange(len(y_te)),y_te]
    covrg =                         v_is_y_te_in_pred.mean()    # mean of vector
    n_y =                           m_is_prs_in_pred_set.shape[1]
    v_covrg_labels =                torch.zeros(n_y, dtype=torch.float)
    v_ineff_labels =                torch.zeros(n_y, dtype=torch.float)
    for i_y in torch.arange(n_y):
        indices_i_y =               (y_te == i_y).nonzero().view(-1)
        v_covrg_labels[i_y] =       m_is_prs_in_pred_set[indices_i_y, y_te[indices_i_y]].mean()
        v_ineff_labels[i_y] =       m_is_prs_in_pred_set[indices_i_y,:].sum(dim=1).mean()
    ineff =                         m_is_prs_in_pred_set.sum(dim=1).mean() # mean of entire matrix
    return covrg, ineff, v_covrg_labels, v_ineff_labels

def kfp_covrg_and_ineff(m_NC_prs, v_NC, y_te, alpha): # prs = prospective   ; size of matrix [N_te,N_prime,N]; of vector [N]
    th =                            (1-alpha)*(len(v_NC)+1)
    m_is_prs_in_pred_set =          ((m_NC_prs > v_NC.view(1,1,-1)).float().sum(dim=2) < th).float()
    v_is_y_te_in_pred =             m_is_prs_in_pred_set[torch.arange(len(y_te)),y_te]
    n_y =                           m_is_prs_in_pred_set.shape[1]
    v_covrg_labels =                torch.zeros(n_y, dtype=torch.float)
    v_ineff_labels =                torch.zeros(n_y, dtype=torch.float)
    for i_y in torch.arange(n_y):
        indices_i_y =               (y_te == i_y).nonzero().view(-1)
        v_covrg_labels[i_y] =       m_is_prs_in_pred_set[indices_i_y, y_te[indices_i_y]].mean()
        v_ineff_labels[i_y] =       m_is_prs_in_pred_set[indices_i_y,:].sum(dim=1).mean()
    covrg =                         v_is_y_te_in_pred.mean()    # mean of vector
    ineff =                         m_is_prs_in_pred_set.sum(dim=1).mean() # mean of entire matrix
    return covrg, ineff, v_covrg_labels, v_ineff_labels
