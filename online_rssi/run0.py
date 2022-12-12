from online_conformal_prediction import *
#import numpy as np # algebra and matrices package
from scipy.io import loadmat, savemat
import torch # Machine learning package
import datetime
import argparse
from copy import deepcopy
import os

parser = argparse.ArgumentParser(description='Conformal Set Prediction')

parser.add_argument('--path_of_run',        dest='path_of_run',         default='run',          type=str)   # path of outputs
parser.add_argument('--dataset_str',        dest='dataset_str',         default='Zanella2014',  type=str)   # ['Zanella2014','Simmons2022']
parser.add_argument('--ind_scenario',       dest='ind_scenario',        default=2,              type=int)   # Zanella2014: {0:'Aisle',1:'Desk',2:'Lab',3:'Outdoor',4:'Room'}
parser.add_argument('--target_miscoverage', dest='target_miscoverage',  default=0.1,            type=float) # [0 to 1]
parser.add_argument('--calibration_scale',  dest='calibration_scale',   default='y',            type=str)   # ['y','quantile'] Feldman's 3.2.1 and 3.2.2
parser.add_argument('--lr',                 dest='lr',                  default=0.01,           type=float) # > 0 # model's learning rate
parser.add_argument('--lr_theta',           dest='lr_theta',            default=0.03,           type=float) # > 0 # theta's learning rate
parser.add_argument('--imbalance_ratio',    dest='imbalance_ratio',     default=0.50,           type=float) # [0 to 1] # 0=sensitive to error_lo; 0.5= balance between error lo and hi; 1=sensitive to error_hi;
parser.add_argument('--opt_weight_decay',   dest='opt_weight_decay',    default=0.1,            type=float) # > 0 # model's L2 reg
parser.add_argument('--seq_len',            dest='seq_len',             default=20,             type=int) # uint  # LSTM sequence length
parser.add_argument('--tr_batch_size',      dest='tr_batch_size',       default=1,              type=int) # uint  # batch size for training, per online sample
parser.add_argument('--num_setup_iters',    dest='num_setup_iters',     default=500,            type=int) # uint  # number of models updates without updating alpha

# Execute the parse_args() method
args =                      parser.parse_args()
torch.manual_seed(          17) # same seed for reproducibility

try:
    os.stat(args.path_of_run)
except:
    os.mkdir(args.path_of_run)
path_of_run =                       args.path_of_run + '/'

startTime =                 datetime.datetime.now()
v_rssi_values =             torch.arange(-95, -26, 1)
alpha_target =              args.target_miscoverage
imbalance_ratio =           args.imbalance_ratio
num_setup_iters =           args.num_setup_iters
lr =                        args.lr
opt_weight_decay =          args.opt_weight_decay
calibration_scale =         args.calibration_scale
lr_theta =                  args.lr_theta
seq_len =                   args.seq_len
if args.dataset_str=='Zanella2014':
    D =                     get_dataset_Zanella2014(args.ind_scenario)
    dim_x =                 D.X.shape[1]
    reg_pred_na_lo =        mlp_lstm_mlp_reg(   dim_x =             dim_x,
                                                f1_hidden_layers =  [16,32],
                                                f2_input_size =     32,
                                                f2_hidden_size =    32,
                                                f2_num_layers =     2,
                                                f3_hidden_layers =  [32])
                                                #scaling_params =    [0.0,1.0,0.0,1.0])#[18.0,16.0,-60.0,80.0])
if args.dataset_str=='Simmons2022':
    D =                     get_dataset_Simmons2022(args.ind_scenario)
    dim_x =                 0
    reg_pred_na_lo =        mlp_lstm_mlp_reg(   dim_x =             dim_x,
                                                f1_hidden_layers =  [16,32],
                                                f2_input_size =     32,
                                                f2_hidden_size =    32,
                                                f2_num_layers =     2,
                                                f3_hidden_layers =  [32])
                                                #scaling_params =    [0.0,1.0,0.0,1.0])#[18.0,16.0,-60.0,80.0])
reg_pred_na_hi =            deepcopy(reg_pred_na_lo)
reg_pred_sy_lo =            deepcopy(reg_pred_na_lo)
reg_pred_sy_hi =            deepcopy(reg_pred_na_lo)
reg_pred_as_lo =            deepcopy(reg_pred_na_lo)
reg_pred_as_hi =            deepcopy(reg_pred_na_lo)
optimizer_na_lo =           torch.optim.Adam(reg_pred_na_lo.parameters(), lr=lr, weight_decay= opt_weight_decay)
optimizer_na_hi =           torch.optim.Adam(reg_pred_na_hi.parameters(), lr=lr, weight_decay= opt_weight_decay)
optimizer_sy_lo =           torch.optim.Adam(reg_pred_sy_lo.parameters(), lr=lr, weight_decay= opt_weight_decay)
optimizer_sy_hi =           torch.optim.Adam(reg_pred_sy_hi.parameters(), lr=lr, weight_decay= opt_weight_decay)
optimizer_as_lo =           torch.optim.Adam(reg_pred_as_lo.parameters(), lr=lr, weight_decay= opt_weight_decay)
optimizer_as_hi =           torch.optim.Adam(reg_pred_as_hi.parameters(), lr=lr, weight_decay= opt_weight_decay)

# accumulate data for adaptation
tr_batch_size =             args.tr_batch_size
i_t_first =                 tr_batch_size+seq_len
N_t =                       len(D)-i_t_first
v_theta_sy =                torch.zeros(N_t+1, dtype=torch.float) # sy = symmetric (like Feldman's)
v_theta_as_lo =             torch.zeros(N_t+1, dtype=torch.float) # lo = low
v_theta_as_hi =             torch.zeros(N_t+1, dtype=torch.float) # hi = high
v_covrg_na =                torch.zeros(N_t  ) # na = naive (without any guarantees for shifts)
v_covrg_sy =                torch.zeros(N_t  )
v_covrg_as =                torch.zeros(N_t  ) # as = asymmetric
v_covrg_as_lo =             torch.zeros(N_t  )
v_covrg_as_hi =             torch.zeros(N_t  )
v_ineff_na =                torch.zeros(N_t  )
v_ineff_sy =                torch.zeros(N_t  )
v_ineff_as =                torch.zeros(N_t  )
v_yhat_t_tr_na_lo =         torch.zeros(N_t  , dtype=torch.float)
v_yhat_t_tr_na_hi =         torch.zeros(N_t  , dtype=torch.float)
v_yhat_t_tr_sy_lo =         torch.zeros(N_t  , dtype=torch.float)
v_yhat_t_tr_sy_hi =         torch.zeros(N_t  , dtype=torch.float)
v_yhat_t_tr_as_lo =         torch.zeros(N_t  , dtype=torch.float)
v_yhat_t_tr_as_hi =         torch.zeros(N_t  , dtype=torch.float)
v_cp_na_lo =                torch.zeros(N_t  , dtype=torch.float)
v_cp_na_hi =                torch.zeros(N_t  , dtype=torch.float)
v_cp_sy_lo =                torch.zeros(N_t  , dtype=torch.float)
v_cp_sy_hi =                torch.zeros(N_t  , dtype=torch.float)
v_cp_as_lo =                torch.zeros(N_t  , dtype=torch.float)
v_cp_as_hi =                torch.zeros(N_t  , dtype=torch.float)
v_stretch_sy =              torch.zeros(N_t  , dtype=torch.float)
v_stretch_as_lo =           torch.zeros(N_t  , dtype=torch.float)
v_stretch_as_hi =           torch.zeros(N_t  , dtype=torch.float)
if calibration_scale=='quantile':
    v_theta_sy[0] =         - alpha_target
    v_theta_as_lo[0] =      - alpha_target
    v_theta_as_hi[0] =      - alpha_target
    stretching_func =       stretching_func_quantile
elif calibration_scale=='y':
    v_theta_sy[0] =         0.0
    v_theta_as_lo[0] =      0.0
    v_theta_as_hi[0] =      0.0
    stretching_func =       stretching_func_exponential
m_loss_na_lo_tr =           torch.zeros(N_t,tr_batch_size)
m_loss_na_hi_tr =           torch.zeros(N_t,tr_batch_size)
m_loss_sy_lo_tr =           torch.zeros(N_t,tr_batch_size)
m_loss_sy_hi_tr =           torch.zeros(N_t,tr_batch_size)
m_loss_as_lo_tr =           torch.zeros(N_t,tr_batch_size)
m_loss_as_hi_tr =           torch.zeros(N_t,tr_batch_size)
for i_t in range(N_t): # index of time
    X_cur, y_cur =          D[i_t_first+i_t         : i_t_first+i_t+1 ]   # we would like to predict y_cur given x_cur
    X_seq, y_seq =          D[i_t_first+i_t-seq_len : i_t_first+i_t   ]   # and given a sequence of previous labeled samples
    with torch.no_grad():
        v_yhat_t_tr_na_lo[i_t] =   reg_pred_na_lo.forward(X_seq, y_seq, X_cur)
        v_yhat_t_tr_na_hi[i_t] =   reg_pred_na_hi.forward(X_seq, y_seq, X_cur)
        v_yhat_t_tr_sy_lo[i_t] =   reg_pred_sy_lo.forward(X_seq, y_seq, X_cur)
        v_yhat_t_tr_sy_hi[i_t] =   reg_pred_sy_hi.forward(X_seq, y_seq, X_cur)
        v_yhat_t_tr_as_lo[i_t] =   reg_pred_as_lo.forward(X_seq, y_seq, X_cur)
        v_yhat_t_tr_as_hi[i_t] =   reg_pred_as_hi.forward(X_seq, y_seq, X_cur)
    v_stretch_sy[i_t] =         stretching_func(v_theta_sy[i_t])
    v_stretch_as_lo[i_t] =      stretching_func(v_theta_as_lo[i_t])
    v_stretch_as_hi[i_t] =      stretching_func(v_theta_as_hi[i_t])
    v_cp_na_lo[i_t] =           v_yhat_t_tr_na_lo[i_t]
    v_cp_na_hi[i_t] =           v_yhat_t_tr_na_hi[i_t]
    v_cp_sy_lo[i_t] =           v_yhat_t_tr_sy_lo[i_t]
    v_cp_sy_hi[i_t] =           v_yhat_t_tr_sy_hi[i_t]
    v_cp_as_lo[i_t] =           v_yhat_t_tr_as_lo[i_t]
    v_cp_as_hi[i_t] =           v_yhat_t_tr_as_hi[i_t]
    if calibration_scale == 'y':                # additive term on y scale. For the other case, the quantile scale, no need to readjust
        v_cp_sy_lo[i_t] -=      v_stretch_sy[i_t]
        v_cp_sy_hi[i_t] +=      v_stretch_sy[i_t]
        v_cp_as_lo[i_t] -=      v_stretch_as_lo[i_t]
        v_cp_as_hi[i_t] +=      v_stretch_as_hi[i_t]
    v_covrg_na[i_t] =       (y_cur>=v_cp_na_lo[i_t]) and (y_cur<=v_cp_na_hi[i_t])
    v_covrg_sy[i_t] =       (y_cur>=v_cp_sy_lo[i_t]) and (y_cur<=v_cp_sy_hi[i_t])
    v_covrg_as[i_t] =       (y_cur>=v_cp_as_lo[i_t]) and (y_cur<=v_cp_as_hi[i_t])
    v_covrg_as_lo[i_t] =    (y_cur>=v_cp_as_lo[i_t])
    v_covrg_as_hi[i_t] =    (y_cur<=v_cp_as_hi[i_t])
    v_ineff_na[i_t] =       v_cp_na_hi[i_t] - v_cp_na_lo[i_t]
    v_ineff_sy[i_t] =       v_cp_sy_hi[i_t] - v_cp_sy_lo[i_t]
    v_ineff_as[i_t] =       v_cp_as_hi[i_t] - v_cp_as_lo[i_t]
    if i_t>=num_setup_iters:
        v_theta_sy[i_t+1] =     (v_theta_sy[i_t]    + lr_theta * ((1-v_covrg_sy[i_t]   ) - alpha_target                       ) )
        v_theta_as_lo[i_t+1] =  (v_theta_as_lo[i_t] + lr_theta * ((1-v_covrg_as_lo[i_t]) - alpha_target * (  imbalance_ratio) ) )
        v_theta_as_hi[i_t+1] =  (v_theta_as_hi[i_t] + lr_theta * ((1-v_covrg_as_hi[i_t]) - alpha_target*  (1-imbalance_ratio) ) )
        if calibration_scale == 'quantile':
            v_theta_sy[   i_t+1].clamp_(  min=-0.99,max=-0.01) # we want to avoid pinball loss with zero slope
            v_theta_as_lo[i_t+1].clamp_(  min=-0.99,max=-0.01) # -0.99 and -0.01 were arbitrarly chsen
            v_theta_as_hi[i_t+1].clamp_(  min=-0.99,max=-0.01)
    else:
        v_theta_sy[i_t+1] =     v_theta_sy[i_t]
        v_theta_as_lo[i_t+1] =  v_theta_as_lo[i_t]
        v_theta_as_hi[i_t+1] =  v_theta_as_hi[i_t]
    # train the model phi_{t+1} using SGD on phi_t
    X_tr, y_tr =            D[ i_t : i_t + i_t_first ]
    for i_tr in range(tr_batch_size):
        X_tr_seq =          X_tr[i_tr : i_tr + seq_len]
        y_tr_seq =          y_tr[i_tr : i_tr + seq_len]
        X_tr_cur =          X_tr[       i_tr + seq_len].view(1,-1)
        y_tr_cur =          y_tr[       i_tr + seq_len].view(1, 1)
        yhat_na_lo =        reg_pred_na_lo.forward(X_tr_seq, y_tr_seq, X_tr_cur)
        yhat_na_hi =        reg_pred_na_hi.forward(X_tr_seq, y_tr_seq, X_tr_cur)
        yhat_sy_lo =        reg_pred_sy_lo.forward(X_tr_seq, y_tr_seq, X_tr_cur)
        yhat_sy_hi =        reg_pred_sy_hi.forward(X_tr_seq, y_tr_seq, X_tr_cur)
        yhat_as_lo =        reg_pred_as_lo.forward(X_tr_seq, y_tr_seq, X_tr_cur)
        yhat_as_hi =        reg_pred_as_hi.forward(X_tr_seq, y_tr_seq, X_tr_cur)
        if calibration_scale=='quantile':
            tau_na_lo =         alpha_target          /2.0
            tau_na_hi =         alpha_target          /2.0
            tau_sy_lo =         v_stretch_sy[i_t]     /2.0
            tau_sy_hi =         v_stretch_sy[i_t]     /2.0
            tau_as_lo =         v_stretch_as_lo[i_t]  /2.0
            tau_as_hi =         v_stretch_as_hi[i_t]  /2.0
        elif calibration_scale=='y':
            tau_na_lo =         alpha_target          /2.0
            tau_na_hi =         alpha_target          /2.0
            tau_sy_lo =         alpha_target          /2.0
            tau_sy_hi =         alpha_target          /2.0
            tau_as_lo =         alpha_target          /2.0
            tau_as_hi =         alpha_target          /2.0
        loss_na_lo_tr =     pinball_loss(yhat_na_lo, y_tr_cur[0],     tau_na_lo)
        loss_na_hi_tr =     pinball_loss(yhat_na_hi, y_tr_cur[0], 1 - tau_na_hi)
        loss_sy_lo_tr =     pinball_loss(yhat_sy_lo, y_tr_cur[0],     tau_sy_lo)
        loss_sy_hi_tr =     pinball_loss(yhat_sy_hi, y_tr_cur[0], 1 - tau_sy_hi)
        loss_as_lo_tr =     pinball_loss(yhat_as_lo, y_tr_cur[0],     tau_as_lo)
        loss_as_hi_tr =     pinball_loss(yhat_as_hi, y_tr_cur[0], 1 - tau_as_hi)
        optimizer_na_lo.zero_grad()
        optimizer_na_hi.zero_grad()
        optimizer_sy_lo.zero_grad()
        optimizer_sy_hi.zero_grad()
        optimizer_as_lo.zero_grad()
        optimizer_as_hi.zero_grad()
        loss_na_lo_tr.backward()
        loss_na_hi_tr.backward()
        loss_sy_lo_tr.backward()
        loss_sy_hi_tr.backward()
        loss_as_lo_tr.backward()
        loss_as_hi_tr.backward()
        optimizer_na_lo.step()
        optimizer_na_hi.step()
        optimizer_sy_lo.step()
        optimizer_sy_hi.step()
        optimizer_as_lo.step()
        optimizer_as_hi.step()
        m_loss_na_lo_tr[i_t,i_tr] =   loss_na_lo_tr.detach()
        m_loss_na_hi_tr[i_t,i_tr] =   loss_na_hi_tr.detach()
        m_loss_sy_lo_tr[i_t,i_tr] =   loss_sy_lo_tr.detach()
        m_loss_sy_hi_tr[i_t,i_tr] =   loss_sy_hi_tr.detach()
        m_loss_as_lo_tr[i_t,i_tr] =   loss_as_lo_tr.detach()
        m_loss_as_hi_tr[i_t,i_tr] =   loss_as_hi_tr.detach()
    print(f'i_t = {i_t}; m_loss_lo_tr= {m_loss_na_lo_tr[i_t,:].mean().item():7.4f}; m_loss_na_hi_tr= {m_loss_na_hi_tr[i_t,:].mean():7.4f}; covrg_sy= {v_covrg_sy[i_t].int():1d}; ineff_sy= {v_ineff_sy[i_t]:.3f};  theta_sy={v_theta_sy[i_t]:5.3f}')
    print(f'i_t = {i_t}; m_loss_lo_tr= {m_loss_na_lo_tr[i_t,:].mean().item():7.4f}; m_loss_na_hi_tr= {m_loss_na_hi_tr[i_t,:].mean():7.4f}; covrg_as= {v_covrg_as[i_t].int():1d}; ineff_as= {v_ineff_as[i_t]:.3f};  theta_as_lo={v_theta_as_lo[i_t]:5.3f}; theta_as_hi={v_theta_as_hi[i_t]:5.3f}')

    if (i_t % 100==0):
        savemat(path_of_run + 'online_rssi_outputs.mat',
                                   {    'D_X':                  D.X.numpy(),
                                        'D_y':                  D.y.numpy(),
                                        'v_theta_sy':           v_theta_sy.numpy(),
                                        'v_theta_as_lo':        v_theta_as_lo.numpy(),
                                        'v_theta_as_hi':        v_theta_as_hi.numpy(),
                                        'v_covrg_na':           v_covrg_na.numpy(),
                                        'v_covrg_sy':           v_covrg_sy.numpy(),
                                        'v_covrg_as':           v_covrg_as.numpy(),
                                        'v_ineff_na':           v_ineff_na.numpy(),
                                        'v_ineff_sy':           v_ineff_sy.numpy(),
                                        'v_ineff_as':           v_ineff_as.numpy(),
                                        'v_stretch_sy':         v_stretch_sy.numpy(),
                                        'v_stretch_as_lo':      v_stretch_as_lo.numpy(),
                                        'v_stretch_as_hi':      v_stretch_as_hi.numpy(),
                                        'v_yhat_t_tr_na_lo':    v_yhat_t_tr_na_lo.numpy(),
                                        'v_yhat_t_tr_na_hi':    v_yhat_t_tr_na_hi.numpy(),
                                        'v_yhat_t_tr_sy_lo':    v_yhat_t_tr_sy_lo.numpy(),
                                        'v_yhat_t_tr_sy_hi':    v_yhat_t_tr_sy_hi.numpy(),
                                        'v_yhat_t_tr_as_lo':    v_yhat_t_tr_as_lo.numpy(),
                                        'v_yhat_t_tr_as_hi':    v_yhat_t_tr_as_hi.numpy(),
                                        'v_cp_na_lo':           v_cp_na_lo.numpy(),
                                        'v_cp_na_hi':           v_cp_na_hi.numpy(),
                                        'v_cp_sy_lo':           v_cp_sy_lo.numpy(),
                                        'v_cp_sy_hi':           v_cp_sy_hi.numpy(),
                                        'v_cp_as_lo':           v_cp_as_lo.numpy(),
                                        'v_cp_as_hi':           v_cp_as_hi.numpy(),
                                        'm_loss_na_lo_tr':      m_loss_na_lo_tr.numpy(),
                                        'm_loss_na_hi_tr':      m_loss_na_hi_tr.numpy(),
                                        'm_loss_sy_lo_tr':      m_loss_sy_lo_tr.numpy(),
                                        'm_loss_sy_hi_tr':      m_loss_sy_hi_tr.numpy(),
                                        'm_loss_as_lo_tr':      m_loss_as_lo_tr.numpy(),
                                        'm_loss_as_hi_tr':      m_loss_as_hi_tr.numpy(),
                                        'v_rssi_values':        v_rssi_values.numpy(),
                                        'lr':                   lr,
                                        'lr_theta':             lr_theta,
                                        'seq_len':              seq_len,
                                        'tr_batch_size':        tr_batch_size,
                                        'imbalance_ratio':      imbalance_ratio,
                                        'num_setup_iters':      num_setup_iters,
                                        'calibration_scale':    calibration_scale,
                                        'alpha_target':         alpha_target,
                                        'i_t_first':            i_t_first,
                                        'i_t':                  i_t,
                                        'num_setup_iters':      num_setup_iters
                                  })