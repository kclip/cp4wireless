#### main
import copy
import pickle
import torch
import sys
import os
import argparse
from vgg16 import VGG16, VGG_small
from datagen_ota import Data_gen_OTA
from general_utils import compute_val_ineff
import NPB # Naive_Probabilistic_Based, GD_for_NPB_mb , Adam_for_NPB, GD_for_NPB
import VB # Split_Conformal, GD_for_VB_mb, GD_for_VB
from XB import CrossVal_plus

print('Start of main.py')

parser = argparse.ArgumentParser(description='Conformal Set Prediction')

parser.add_argument('--set_prediction_mode',    dest='set_prediction_mode', default='NPB',      type=str)   # NPB VB CV+4 CV+40
parser.add_argument('--alpha',                  dest='alpha',               default=0.1,        type=float) # [0 to 1]
parser.add_argument('--num_classes',            dest='num_classes',         default=24,         type=int)   #
parser.add_argument('--N_over_num_classes',     dest='N_over_num_classes',  default=150,        type=int)   #
parser.add_argument('--lr_inner',               dest='lr_inner',            default=0.020,      type=float) # > 0 # model's learning rate
parser.add_argument('--num_indep_trial',        dest='num_indep_trial',     default=32,         type=int)   # num of independent trials
parser.add_argument('--N_te',                   dest='N_te',                default=1000,       type=int)   # num of test points
parser.add_argument('--inner_iter',             dest='inner_iter',          default=4000,       type=int)   # num of test points
parser.add_argument('--num_of_minibatches',     dest='num_of_minibatches',  default=1,          type=int)   # num of test points
parser.add_argument('--snr_db_th',              dest='snr_db_th',           default=6,          type=int)   # snr in dB threshold for data set

args =                      parser.parse_args()

##### basic set-up
set_prediction_mode =       args.set_prediction_mode
alpha =                     args.alpha
num_classes =               args.num_classes
lr_inner =                  args.lr_inner
num_of_minibatches =        args.num_of_minibatches
snr_db_th =                 args.snr_db_th
latch_best_tr =             True
if sys.platform.startswith('linux'): # LINUX
    args.cuda_ind =         0
    args.device =           torch.device("cuda:" + str(args.cuda_ind) if torch.cuda.is_available() else "cpu")
    set_prediction_mode =   args.set_prediction_mode
    N =                     num_classes * args.N_over_num_classes
    num_indep_trial =       args.num_indep_trial
    N_te =                  args.N_te
    inner_iter =            args.inner_iter
    xi = VGG16().to(args.device)                # define init. of NN -- currently generated once, and used for all experiments
else: # WINDOWS
    print('OS not linux: so we debug using reduced network (not VGG) and small training size')
    args.device =           torch.device("cuda:" + str(args.cuda_ind) if torch.cuda.is_available() else "cpu")
    set_prediction_mode =   'CV+4' # 'CV+4' # 'NPB' # 'VB' # f'CV+{num_classes}' # VB
    print(f'forcing set_prediction_mode to {set_prediction_mode =}')
    N =                     num_classes * min(4,args.N_over_num_classes)   #  not too many for windows...
    num_indep_trial =       5
    N_te =                  100
    inner_iter =            20
    snr_db_th =             24 # less vectors in training
    xi = VGG_small().to(args.device)            # define init. of NN -- currently generated once, and used for all experiments

# define data generator
print(f'running set_prediction_mode={set_prediction_mode} in device = {args.device}')
print('Loading data')
datagen_modulation_classification = Data_gen_OTA(N+N_te,snr_db_th)
print('Finished loading data')
N_tr_small = 10000 # avoid GPU out of memory for gauging training loss
try:
    os.stat('saved')
except:
    os.mkdir('saved')
    print('Making new directory saved/')

if set_prediction_mode == 'NPB': # naive probabilistic based
    print('constructing Naive_Probabilistic_Based')
    NPB_inst =                          NPB.Naive_Probabilistic_Based(
                                        alpha=          alpha,
                                        num_classes=    num_classes)
elif set_prediction_mode == 'VB': # validation-based = split conformal
    print('constructing Split_Conformal')
    VB_inst =                           VB.Split_Conformal(
                                        alpha=          alpha,
                                        num_classes=    num_classes)
elif 'CV+' in set_prediction_mode:
    print('constructing CrossVal_plus')
    # 'CV+5' 5 stands for K  -- i.e., 'CV+K'
    CV_plus =                       CrossVal_plus(
                                        K=              int(set_prediction_mode[3:]),
                                        alpha=          alpha,
                                        xi=             xi,
                                        num_classes=    num_classes,
                                        lr_inner=       lr_inner,
                                        inner_iter=     inner_iter,
                                        if_JK_mm=       'mm' in set_prediction_mode,
                                        latch_best_tr=  latch_best_tr)
else:
    raise NotImplementedError

##### actual run
dict_for_pd =                           {}
dict_for_pd['Accuracy tr'] =            [] # only for NPB and VB, this gauges tr accuracy of the single learnt model
dict_for_pd['Accuracy te'] =            [] # only for NPB and VB, this gauges te accuracy of the single learnt model
dict_for_pd['Coverage'] =               []
dict_for_pd['Inefficiency'] =           []
dict_for_pd['Set prediction mode'] =    []
eval_dict_save_path_for_pkl =           './saved/'+ 'alpha_' + str(alpha) + 'N_' + str(N)+ 'set_predictor_' + set_prediction_mode + 'lr_inner_' + str(lr_inner) + 'inner_iter' + str(inner_iter) + 'dataset_for_box_plot'
sum_coverage =                          0
sum_ineff =                             0
sum_accuracy_tr =                       0
sum_accuracy_te =                       0
for indep_trial in range(num_indep_trial):
    print(f'indep trial {indep_trial}/{num_indep_trial}: generating tr+te data with N={N} pairs')
    tr_dataset, te_dataset =            datagen_modulation_classification.gen(N,device=args.device)
    X_tr,Y_tr =                         tr_dataset[0],tr_dataset[1]
    X_te,Y_te =                         te_dataset[0], te_dataset[1]
    print(f'indep trial {indep_trial}/{num_indep_trial}: data generated')
    if set_prediction_mode == 'NPB':
        phi_npb =                       copy.deepcopy(xi).to(args.device)
        phi_npb =                       phi_npb.to(args.device) # initialization    npb = naive probabilistic-based
        print(f'indep trial {indep_trial}/{num_indep_trial} NPB GD_for_NPB')
        phi_npb =                       NPB.GD_for_NPB( phi_npb,
                                                    tr_dataset,
                                                    lr_inner,
                                                    inner_iter,
                                                    num_of_minibatches,
                                                    latch_best_tr)  # training using the full available data set:
        print(f'indep trial {indep_trial}/{num_indep_trial} NPB forward')
        pred_set_as_indicator_matrix =  NPB_inst.forward(phi_npb,
                                                    None,
                                                    X_te)
        m_prob_npb_y_tr =               phi_npb(    X_tr[0:N_tr_small],
                                                    None,
                                                    if_softmax_out=True)
        accuracy_tr =                   (m_prob_npb_y_tr.argmax(dim=1) == Y_tr[0:N_tr_small].squeeze()).type(dtype=torch.float).mean().item()
        m_prob_npb_y_te =               phi_npb(    X_te,
                                                    None,
                                                    if_softmax_out=True)
        accuracy_te =                   (m_prob_npb_y_te.argmax(dim=1) == Y_te.squeeze()).type(dtype=torch.float).mean().item()
    elif set_prediction_mode == 'VB':
        # divide tr_dataset into two parts -- one for ConfTR (as proper training set) the other for calibration
        proper_tr_dataset =             (X_tr[:N//2], Y_tr[:N//2])
        cali_tr_dataset =               (X_tr[N//2:], Y_tr[N//2:])
        phi_vb_ =                       copy.deepcopy(xi).to(args.device)
        print(f'indep trial {indep_trial}/{num_indep_trial} VB GD_for_VB')
        phi_vb_ =                       VB.GD_for_VB(  phi_vb_,
                                                    proper_tr_dataset,
                                                    lr_inner,
                                                    inner_iter,
                                                    num_of_minibatches,
                                                    latch_best_tr)  # training using only the proper part of the available data set
        print(f'indep trial {indep_trial}/{num_indep_trial} VB forward')
        pred_set_as_indicator_matrix =  VB_inst.forward( phi_vb_,
                                                    cali_tr_dataset,
                                                    X_te)
        m_prob_vb__y_tr =               phi_vb_(    proper_tr_dataset[0][0:N_tr_small],
                                                    None,
                                                    if_softmax_out=True)
        accuracy_tr =                   (m_prob_vb__y_tr.argmax(dim=1) == proper_tr_dataset[1][0:N_tr_small].squeeze()).type(dtype=torch.float).mean().item()
        m_prob_vb__y_te =               phi_vb_(    X_te,
                                                    None,
                                                    if_softmax_out=True)
        accuracy_te =                   (m_prob_vb__y_te.argmax(dim=1) == Y_te.squeeze()).type(dtype=torch.float).mean().item()
    elif 'CV+' in set_prediction_mode:
        print(f'indep trial {indep_trial}/{num_indep_trial} CV forward')
        pred_set_as_indicator_matrix =  CV_plus.forward(tr_dataset,
                                                        X_te)
        accuracy_tr =                   float(torch.nan)
        accuracy_te =                   float(torch.nan)
    else:
        raise NotImplementedError
    curr_validity_unnorm, \
    curr_inefficiency_unnorm =          compute_val_ineff(pred_set_as_indicator_matrix, Y_te)
    curr_validity =                     float(curr_validity_unnorm/X_te.shape[0])
    curr_ineff =                        float(curr_inefficiency_unnorm/X_te.shape[0])
    dict_for_pd['Accuracy tr'].append(          accuracy_tr)
    dict_for_pd['Accuracy te'].append(          accuracy_te)
    print(f'accuracy_tr = {accuracy_tr}  accuracy_te = {accuracy_te}')
    dict_for_pd['Coverage'].append(    curr_validity)
    dict_for_pd['Inefficiency'].append(         curr_ineff)
    dict_for_pd['Set prediction mode'].append(  set_prediction_mode)
    sum_coverage +=                     curr_validity
    sum_ineff +=                        curr_ineff
    sum_accuracy_tr +=                  accuracy_tr
    sum_accuracy_te +=                  accuracy_te
    print('N: ', N, 'alpha: ', alpha, 'indep_trial: ', indep_trial, '    coverage:          ', curr_validity               , '    ineff:          ', curr_ineff               ,'     accuracy_te:          ',accuracy_te                    ,'     accuracy_tr:          ',accuracy_tr                    )
    print('N: ', N, 'alpha: ', alpha, 'indep_trial: ', indep_trial, 'avg coverage upto now: ', sum_coverage/(indep_trial+1), 'avg ineff upto now: ', sum_ineff/(indep_trial+1),' avg accuracy_te upto now: ',sum_accuracy_te/(indep_trial+1),' avg accuracy_tr upto now: ',sum_accuracy_tr/(indep_trial+1))
    # offload products to memory
    with open(eval_dict_save_path_for_pkl, 'wb') as f:
        print(f'indep trial {indep_trial}/{num_indep_trial} pickle dumping')
        print('')
        print('')
        print('')
        pickle.dump(dict_for_pd, f)
