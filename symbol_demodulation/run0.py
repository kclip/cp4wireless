from conformal_prediction import *
import numpy as np # algebra and matrices package
from scipy.io import loadmat, savemat
import torch # Machine learning package
import datetime
import argparse
import os

parser = argparse.ArgumentParser(description='Conformal Set Prediction')

parser.add_argument('--path_of_run',    dest='path_of_run',     default='run',      type=str)
parser.add_argument('--snr_dB',         dest='snr_dB',          default=5.0,        type=float)
parser.add_argument('--modKey',         dest='modKey',          default='8APSK',    type=str)   # modulation key. QPSK / 8APSK / 16QAM/ 64QAM / 8GAM / 16GAM
parser.add_argument('--i_gamma',        dest='i_gamma',         default=8,          type=int)   # regularization coeff for MAP
parser.add_argument('--gd_lr',          dest='gd_lr',           default=2e-1,       type=float) # Learning rate while GDing
parser.add_argument('--gd_num_iters',   dest='gd_num_iters',    default=120,        type=int)   # Number of GD iterations (for ML and MAP)
parser.add_argument('--lmc_lr_init',    dest='lmc_lr_init',     default=2e-1,       type=float) # Learning rate while GDing
parser.add_argument('--lmc_lr_decaying',dest='lmc_lr_decaying', default=1.0,        type=float) # Learning rate decaying factor over iterations
parser.add_argument('--lmc_burn_in',    dest='lmc_burn_in',     default=100,        type=int)   # total number of burn in first discarded model parameters while LMCing
parser.add_argument('--ensemble_size'  ,dest='ensemble_size',   default=20,         type=int)   # Ensemble prediction for Bayesian methods
parser.add_argument('--num_sim'        ,dest='num_sim',         default=50,         type=int)   # 10? number of independent simulation runs
parser.add_argument('--compute_hessian',dest='compute_hessian', default=0,          type=int)   #{0: all zero hessian = faster = nonsense, 1: model-based using external network, 2: grad(grad) model agnoistic}

# Execute the parse_args() method
args =                              parser.parse_args()
torch.manual_seed(                  0) # same seed for reproducibility

startTime =                         datetime.datetime.now()
dSetting =                          {   'snr_dB'     : args.snr_dB,  # Signal to noise ratio per one RX antenna
                                        'modKey'     : args.modKey } # modulation key
tPercentiles =                      (90, 50)
iotUplink =                         IotUplink(dSetting)
FirstLayerDim =                     2  # I and Q
LastLayerDim =                      iotUplink.modulator.order  # num of classes is the total QAM symbols
vLayers =                           [FirstLayerDim, 10, 30, 30, LastLayerDim]
gd_num_iters =                      args.gd_num_iters
gd_lr =                             args.gd_lr
ensemble_size =                     args.ensemble_size # 50 Ensemble prediction for Bayesian methods
compute_hessian =                   args.compute_hessian

K =                                 4      # K-fold for cross-validation
v_N   =                             np.arange(12,48+1,K) # Data set size                                                 #np.arange(12,28+1,K) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
num_N =                             len(v_N)
num_sim =                           args.num_sim
N_te =                              LastLayerDim*12     # 100 # number of test points
alpha =                             0.1    # miscoverage level
alpha_cvb =                         alpha  # miscoverage level for cross-validation-based. Can be alpha (for weak guarantee) or alpha/2 (for strong gaurantee)
N_prime =                           LastLayerDim
Y_prime =                           torch.tensor(range(N_prime))
v_gamma =                           [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1e0, 2e0, 5e0]
#                                    0     1     2     3     4     5     6     7     8     9     10    11    12   13   14
gamma =                             v_gamma[args.i_gamma]

lmc_burn_in =                       args.lmc_burn_in
lmc_lr_init =                       args.lmc_lr_init
lmc_lr_decaying =                   args.lmc_lr_decaying

i_ml_ =                             0
i_map =                             1
i_hes =                             2
i_fim =                             3
i_lmc =                             4
v_alg_str =                         ['frequentist ML','frequentist MAP','Bayesian using Hessian','Bayesian using FIM','Bayesian using LMC']

m_malloc3d =                        np.empty((num_N,len(v_alg_str),num_sim))
m_malloc4d =                        np.empty((num_N,len(v_alg_str),num_sim,LastLayerDim))
# In the following, s for struct  covrg = coverage  ineff = width
s_sft =                             {'covrg': m_malloc3d.copy(), 'ineff': m_malloc3d.copy(), 'covrg_labels': m_malloc4d.copy(), 'ineff_labels': m_malloc4d.copy()}  # sft = soft (non valid)
s_vb_ =                             {'covrg': m_malloc3d.copy(), 'ineff': m_malloc3d.copy(), 'covrg_labels': m_malloc4d.copy(), 'ineff_labels': m_malloc4d.copy()}  # vb_ = validation based (split)
s_vb__giq =                         {'covrg': m_malloc3d.copy(), 'ineff': m_malloc3d.copy(), 'covrg_labels': m_malloc4d.copy(), 'ineff_labels': m_malloc4d.copy()}  # vb_ = validation based (split)
s_jkp =                             {'covrg': m_malloc3d.copy(), 'ineff': m_malloc3d.copy(), 'covrg_labels': m_malloc4d.copy(), 'ineff_labels': m_malloc4d.copy()}  # jkp = cross validation+ jackknife+
s_jkp_giq =                         {'covrg': m_malloc3d.copy(), 'ineff': m_malloc3d.copy(), 'covrg_labels': m_malloc4d.copy(), 'ineff_labels': m_malloc4d.copy()}  # jkp = cross validation+ jackknife+
s_kfp =                             {'covrg': m_malloc3d.copy(), 'ineff': m_malloc3d.copy(), 'covrg_labels': m_malloc4d.copy(), 'ineff_labels': m_malloc4d.copy()}  # kfp = cross validation+ K-fold jackknife+
s_kfp_giq =                         {'covrg': m_malloc3d.copy(), 'ineff': m_malloc3d.copy(), 'covrg_labels': m_malloc4d.copy(), 'ineff_labels': m_malloc4d.copy()}  # kfp = cross validation+ K-fold jackknife+

try:
    os.stat(args.path_of_run)
except:
    os.mkdir(args.path_of_run)
path_of_run =                       args.path_of_run + '/'


print_str =                         ''

m_loss_tr_sft_ml_ =                 torch.zeros((num_sim, num_N, 1,         gd_num_iters), dtype = torch.float)
m_loss_te_sft_ml_ =                 torch.zeros((num_sim, num_N, 1,         gd_num_iters), dtype = torch.float)
m_loss_tr_sft_map =                 torch.zeros((num_sim, num_N, 1,         gd_num_iters), dtype = torch.float)
m_loss_te_sft_map =                 torch.zeros((num_sim, num_N, 1,         gd_num_iters), dtype = torch.float)
m_loss_tr_vb__ml_ =                 torch.zeros((num_sim, num_N, 1,         gd_num_iters), dtype = torch.float)
m_loss_te_vb__ml_ =                 torch.zeros((num_sim, num_N, 1,         gd_num_iters), dtype = torch.float)
m_loss_tr_vb__map =                 torch.zeros((num_sim, num_N, 1,         gd_num_iters), dtype = torch.float)
m_loss_te_vb__map =                 torch.zeros((num_sim, num_N, 1,         gd_num_iters), dtype = torch.float)
m_loss_tr_jkp_ml_ =                 torch.zeros((num_sim, num_N, v_N.max(), gd_num_iters), dtype = torch.float)
m_loss_te_jkp_ml_ =                 torch.zeros((num_sim, num_N, v_N.max(), gd_num_iters), dtype = torch.float)
m_loss_tr_jkp_map =                 torch.zeros((num_sim, num_N, v_N.max(), gd_num_iters), dtype = torch.float)
m_loss_te_jkp_map =                 torch.zeros((num_sim, num_N, v_N.max(), gd_num_iters), dtype = torch.float)
m_loss_tr_kfp_ml_ =                 torch.zeros((num_sim, num_N, K,         gd_num_iters), dtype = torch.float)
m_loss_te_kfp_ml_ =                 torch.zeros((num_sim, num_N, K,         gd_num_iters), dtype = torch.float)
m_loss_tr_kfp_map =                 torch.zeros((num_sim, num_N, K,         gd_num_iters), dtype = torch.float)
m_loss_te_kfp_map =                 torch.zeros((num_sim, num_N, K,         gd_num_iters), dtype = torch.float)

lmc_num_iters =                     lmc_burn_in + ensemble_size - 1
m_loss_tr_sft_lmc =                 torch.zeros((num_sim, num_N, 1,         lmc_num_iters), dtype = torch.float)
m_loss_te_sft_lmc =                 torch.zeros((num_sim, num_N, 1,         lmc_num_iters), dtype = torch.float)
m_loss_tr_vb__lmc =                 torch.zeros((num_sim, num_N, 1,         lmc_num_iters), dtype = torch.float)
m_loss_te_vb__lmc =                 torch.zeros((num_sim, num_N, 1,         lmc_num_iters), dtype = torch.float)
m_loss_tr_jkp_lmc =                 torch.zeros((num_sim, num_N, v_N.max(), lmc_num_iters), dtype = torch.float)
m_loss_te_jkp_lmc =                 torch.zeros((num_sim, num_N, v_N.max(), lmc_num_iters), dtype = torch.float)
m_loss_tr_kfp_lmc =                 torch.zeros((num_sim, num_N, K,         lmc_num_iters), dtype = torch.float)
m_loss_te_kfp_lmc =                 torch.zeros((num_sim, num_N, K,         lmc_num_iters), dtype = torch.float)


itr_file_str = path_of_run+'iterations.txt'
try:
    os.remove(itr_file_str)
except OSError:
    pass

for i_s in range(num_sim):
    itr_file = open(itr_file_str, 'a')
    itr_file.write(f"iter {i_s} of {num_sim}. time {datetime.datetime.now().strftime('%H:%M:%S')}.\n")
    itr_file.close() # for updating the file instantly
    sim_iter_str =          f" sim ={i_s}/{num_sim-1}"
    print(print_str + sim_iter_str)
    iotUplink.draw_channel_state()                  # new channel state, groundtruth c
    D_full =                                iotUplink.step(v_N.max(),   False, False)
    D_te  =                                 iotUplink.step(N_te,        False, False)
    net_init =                              FcReluDnn(vLayers)  # draw by random
    gd_init_sd =                            net_init.state_dict()
    for i_N in range(num_N):
        sweep_iter_str = f" sweep ={i_N}/{num_N-1}"
        print(print_str + sim_iter_str + sweep_iter_str)
        N =                                 v_N[i_N]
        D,_ =                               D_full.split_into_two_subsets(N)
        ## sft = soft
        print(print_str + sim_iter_str + sweep_iter_str+' sft')
        D_sft =                             D
        model_sft_ml_ =                     deepcopy(               net_init) # important this has the same init vector like all others
        model_sft_map =                     deepcopy(               net_init) # "
        model_sft_ens =                     deepcopy(               net_init) # "
        m_loss_tr_sft_ml_[i_s,i_N,0,:],\
            m_loss_te_sft_ml_[i_s,i_N,0,:]= fitting_erm_ml__gd(     model_sft_ml_, D_sft, D_te, gd_init_sd, gd_lr, gd_num_iters                          )   # ML
        m_loss_tr_sft_map[i_s,i_N,0,:],\
            m_loss_te_sft_map[i_s, i_N,0,:],\
            m_loss_tr_sft_lmc[i_s,i_N,0,:],\
            m_loss_te_sft_lmc[i_s, i_N,0,:],\
            ens_sft_hes,\
            ens_sft_fim,\
            ens_sft_lmc=                    fitting_erm_map_gd(     model_sft_map,
                                                                    D_sft, 
                                                                    D_te, 
                                                                    gd_init_sd, 
                                                                    gd_lr, 
                                                                    gd_num_iters, 
                                                                    gamma, 
                                                                    ensemble_size,
                                                                    compute_hessian,
                                                                    lmc_burn_in,
                                                                    lmc_lr_init,
                                                                    lmc_lr_decaying)   # MAP
        m_prob_y_te__sft_ml_  =             torch.nn.functional.softmax(    model_sft_ml_( D_te.X   ), dim=1)  # prediction
        m_prob_y_te__sft_map  =             torch.nn.functional.softmax(    model_sft_map( D_te.X   ), dim=1)  # prediction
        m_prob_y_te__sft_hes  =             ensemble_predict(               D_te.X ,  ens_sft_hes, model_sft_ens)
        m_prob_y_te__sft_fim  =             ensemble_predict(               D_te.X ,  ens_sft_fim, model_sft_ens)
        m_prob_y_te__sft_lmc  =             ensemble_predict(               D_te.X ,  ens_sft_lmc, model_sft_ens)
        s_sft['covrg'][i_N,i_ml_,i_s],s_sft['ineff'][i_N,i_ml_,i_s],s_sft['covrg_labels'][i_N,i_ml_,i_s,:],s_sft['ineff_labels'][i_N,i_ml_,i_s,:] =   sft_covrg_and_ineff(m_prob_y_te__sft_ml_, D_te.y , alpha)
        s_sft['covrg'][i_N,i_map,i_s],s_sft['ineff'][i_N,i_map,i_s],s_sft['covrg_labels'][i_N,i_map,i_s,:],s_sft['ineff_labels'][i_N,i_map,i_s,:] =   sft_covrg_and_ineff(m_prob_y_te__sft_map, D_te.y , alpha)
        s_sft['covrg'][i_N,i_hes,i_s],s_sft['ineff'][i_N,i_hes,i_s],s_sft['covrg_labels'][i_N,i_hes,i_s,:],s_sft['ineff_labels'][i_N,i_hes,i_s,:] =   sft_covrg_and_ineff(m_prob_y_te__sft_hes, D_te.y , alpha)
        s_sft['covrg'][i_N,i_fim,i_s],s_sft['ineff'][i_N,i_fim,i_s],s_sft['covrg_labels'][i_N,i_fim,i_s,:],s_sft['ineff_labels'][i_N,i_fim,i_s,:] =   sft_covrg_and_ineff(m_prob_y_te__sft_fim, D_te.y , alpha)
        s_sft['covrg'][i_N,i_lmc,i_s],s_sft['ineff'][i_N,i_lmc,i_s],s_sft['covrg_labels'][i_N,i_lmc,i_s,:],s_sft['ineff_labels'][i_N,i_lmc,i_s,:] =   sft_covrg_and_ineff(m_prob_y_te__sft_lmc, D_te.y , alpha)

        ## vb_ = validation based (split)
        print(print_str + sim_iter_str + sweep_iter_str+' vb_')
        N_tr =                              int(N/2) #max(d,floor(N/2))                      # num of proper training points
        N_val =                             N - N_tr
        D_vb__tr,D_vb__val =                D.split_into_two_subsets(N_tr,False)
        model_vb__ml_ =                     deepcopy(               net_init) # important this has the same init vector like all others
        model_vb__map =                     deepcopy(               net_init) # "
        model_vb__ens =                     deepcopy(               net_init) # to be used in the Bayes, changes every call of nonconformity_bay
        m_loss_tr_vb__ml_[i_s,i_N,0,:],\
            m_loss_te_vb__ml_[i_s,i_N,0,:] =fitting_erm_ml__gd(     model_vb__ml_, D_vb__tr, D_te, gd_init_sd, gd_lr, gd_num_iters                          )   # ML
        m_loss_tr_vb__map[i_s,i_N,0,:],\
            m_loss_te_vb__map[i_s,i_N,0,:], \
            m_loss_tr_vb__lmc[i_s, i_N, 0, :], \
            m_loss_te_vb__lmc[i_s, i_N, 0, :], \
            ens_vb__hes,\
            ens_vb__fim, \
            ens_vb__lmc =                   fitting_erm_map_gd(     model_vb__map, 
                                                                    D_vb__tr, 
                                                                    D_te, 
                                                                    gd_init_sd, 
                                                                    gd_lr, 
                                                                    gd_num_iters, 
                                                                    gamma, 
                                                                    ensemble_size,
                                                                    compute_hessian,
                                                                    lmc_burn_in,
                                                                    lmc_lr_init,
                                                                    lmc_lr_decaying )   # MAP
        v_NC_vb__ml_ =                      nonconformity_frq(      D_vb__val.X, D_vb__val.y, model_vb__ml_)
        v_NC_vb__map =                      nonconformity_frq(      D_vb__val.X, D_vb__val.y, model_vb__map)
        v_NC_vb__hes =                      nonconformity_bay(      D_vb__val.X, D_vb__val.y, model_vb__ens, ens_vb__hes)
        v_NC_vb__fim =                      nonconformity_bay(      D_vb__val.X, D_vb__val.y, model_vb__ens, ens_vb__fim)
        v_NC_vb__lmc =                      nonconformity_bay(      D_vb__val.X, D_vb__val.y, model_vb__ens, ens_vb__lmc)
        v_NC_vb__ml__giq =                  nonconformity_frq_giq(  D_vb__val.X, D_vb__val.y, model_vb__ml_)
        v_NC_vb__map_giq =                  nonconformity_frq_giq(  D_vb__val.X, D_vb__val.y, model_vb__map)
        v_NC_vb__hes_giq =                  nonconformity_bay_giq(  D_vb__val.X, D_vb__val.y, model_vb__ens, ens_vb__hes)
        v_NC_vb__fim_giq =                  nonconformity_bay_giq(  D_vb__val.X, D_vb__val.y, model_vb__ens, ens_vb__fim)
        v_NC_vb__lmc_giq =                  nonconformity_bay_giq(  D_vb__val.X, D_vb__val.y, model_vb__ens, ens_vb__lmc)
        X_pairs =                           D_te.X.repeat_interleave(len(Y_prime),dim=0)
        y_pairs =                           Y_prime.repeat(D_te.X.shape[0])
        N_pairs =                           N_te*N_prime
        m_NC_vb__ml__prs =                  nonconformity_frq(      X_pairs, y_pairs, model_vb__ml_             ).view(N_te,N_prime)
        m_NC_vb__map_prs =                  nonconformity_frq(      X_pairs, y_pairs, model_vb__map             ).view(N_te,N_prime)
        m_NC_vb__hes_prs =                  nonconformity_bay(      X_pairs, y_pairs, model_vb__ens, ens_vb__hes).view(N_te,N_prime)
        m_NC_vb__fim_prs =                  nonconformity_bay(      X_pairs, y_pairs, model_vb__ens, ens_vb__fim).view(N_te,N_prime)
        m_NC_vb__lmc_prs =                  nonconformity_bay(      X_pairs, y_pairs, model_vb__ens, ens_vb__lmc).view(N_te,N_prime)
        m_NC_vb__ml__prs_giq =              nonconformity_frq_giq(  X_pairs, y_pairs, model_vb__ml_             ).view(N_te,N_prime)
        m_NC_vb__map_prs_giq =              nonconformity_frq_giq(  X_pairs, y_pairs, model_vb__map             ).view(N_te,N_prime)
        m_NC_vb__hes_prs_giq =              nonconformity_bay_giq(  X_pairs, y_pairs, model_vb__ens, ens_vb__hes).view(N_te,N_prime)
        m_NC_vb__fim_prs_giq =              nonconformity_bay_giq(  X_pairs, y_pairs, model_vb__ens, ens_vb__fim).view(N_te,N_prime)
        m_NC_vb__lmc_prs_giq =              nonconformity_bay_giq(  X_pairs, y_pairs, model_vb__ens, ens_vb__lmc).view(N_te,N_prime)
        s_vb_    ['covrg'][i_N,i_ml_,i_s],s_vb_    ['ineff'][i_N,i_ml_,i_s],s_vb_    ['covrg_labels'][i_N,i_ml_,i_s,:],s_vb_    ['ineff_labels'][i_N,i_ml_,i_s,:] =   vb__covrg_and_ineff(m_NC_vb__ml__prs    , v_NC_vb__ml_    , D_te.y, alpha)
        s_vb_    ['covrg'][i_N,i_map,i_s],s_vb_    ['ineff'][i_N,i_map,i_s],s_vb_    ['covrg_labels'][i_N,i_map,i_s,:],s_vb_    ['ineff_labels'][i_N,i_map,i_s,:] =   vb__covrg_and_ineff(m_NC_vb__map_prs    , v_NC_vb__map    , D_te.y, alpha)
        s_vb_    ['covrg'][i_N,i_hes,i_s],s_vb_    ['ineff'][i_N,i_hes,i_s],s_vb_    ['covrg_labels'][i_N,i_hes,i_s,:],s_vb_    ['ineff_labels'][i_N,i_hes,i_s,:] =   vb__covrg_and_ineff(m_NC_vb__hes_prs    , v_NC_vb__hes    , D_te.y, alpha)
        s_vb_    ['covrg'][i_N,i_fim,i_s],s_vb_    ['ineff'][i_N,i_fim,i_s],s_vb_    ['covrg_labels'][i_N,i_fim,i_s,:],s_vb_    ['ineff_labels'][i_N,i_fim,i_s,:] =   vb__covrg_and_ineff(m_NC_vb__fim_prs    , v_NC_vb__fim    , D_te.y, alpha)
        s_vb_    ['covrg'][i_N,i_lmc,i_s],s_vb_    ['ineff'][i_N,i_lmc,i_s],s_vb_    ['covrg_labels'][i_N,i_lmc,i_s,:],s_vb_    ['ineff_labels'][i_N,i_lmc,i_s,:] =   vb__covrg_and_ineff(m_NC_vb__lmc_prs    , v_NC_vb__lmc    , D_te.y, alpha)
        s_vb__giq['covrg'][i_N,i_ml_,i_s],s_vb__giq['ineff'][i_N,i_ml_,i_s],s_vb__giq['covrg_labels'][i_N,i_ml_,i_s,:],s_vb__giq['ineff_labels'][i_N,i_ml_,i_s,:] =   vb__covrg_and_ineff(m_NC_vb__ml__prs_giq, v_NC_vb__ml__giq, D_te.y, alpha)
        s_vb__giq['covrg'][i_N,i_map,i_s],s_vb__giq['ineff'][i_N,i_map,i_s],s_vb__giq['covrg_labels'][i_N,i_map,i_s,:],s_vb__giq['ineff_labels'][i_N,i_map,i_s,:] =   vb__covrg_and_ineff(m_NC_vb__map_prs_giq, v_NC_vb__map_giq, D_te.y, alpha)
        s_vb__giq['covrg'][i_N,i_hes,i_s],s_vb__giq['ineff'][i_N,i_hes,i_s],s_vb__giq['covrg_labels'][i_N,i_hes,i_s,:],s_vb__giq['ineff_labels'][i_N,i_hes,i_s,:] =   vb__covrg_and_ineff(m_NC_vb__hes_prs_giq, v_NC_vb__hes_giq, D_te.y, alpha)
        s_vb__giq['covrg'][i_N,i_fim,i_s],s_vb__giq['ineff'][i_N,i_fim,i_s],s_vb__giq['covrg_labels'][i_N,i_fim,i_s,:],s_vb__giq['ineff_labels'][i_N,i_fim,i_s,:] =   vb__covrg_and_ineff(m_NC_vb__fim_prs_giq, v_NC_vb__fim_giq, D_te.y, alpha)
        s_vb__giq['covrg'][i_N,i_lmc,i_s],s_vb__giq['ineff'][i_N,i_lmc,i_s],s_vb__giq['covrg_labels'][i_N,i_lmc,i_s,:],s_vb__giq['ineff_labels'][i_N,i_lmc,i_s,:] =   vb__covrg_and_ineff(m_NC_vb__lmc_prs_giq, v_NC_vb__lmc_giq, D_te.y, alpha)

        ## jkp = jackknife+
        print(print_str + sim_iter_str + sweep_iter_str + ' jkp fitting')
        l_model_jkp_ml_ =               []
        l_model_jkp_map =               []
        l_ens_jkp_hes =                 []
        l_ens_jkp_fim =                 []
        l_ens_jkp_lmc =                 []
        for i in range(N):
            sweep_i_str =                               f" sample ={i}/{N - 1}"
            print(print_str + sim_iter_str + sweep_iter_str + ' jkp fitting' + sweep_i_str)
            model_jkp_ml_ =                             deepcopy(               net_init) # important this has the same init vector like all others
            model_jkp_map =                             deepcopy(               net_init) # "
            model_jkp_ens =                             deepcopy(               net_init) # to be used in the Bayes, changes every call of nonconformity_bay
            D_loo =                                     D.leave_one_out(i)
            m_loss_tr_jkp_ml_[i_s,i_N,i,:],\
                m_loss_te_jkp_ml_[i_s,i_N,i,:] =        fitting_erm_ml__gd(     model_jkp_ml_, D_loo, D_te, gd_init_sd, gd_lr, gd_num_iters                          )   # ML
            m_loss_tr_jkp_map[i_s,i_N,i,:],\
                m_loss_te_jkp_map[i_s,i_N,i,:], \
                m_loss_tr_jkp_lmc[i_s, i_N, i, :], \
                m_loss_te_jkp_lmc[i_s, i_N, i, :], \
                ens_jkp_hes,\
                ens_jkp_fim,\
                ens_jkp_lmc =                          fitting_erm_map_gd(     model_jkp_map, 
                                                                                D_loo, 
                                                                                D_te, 
                                                                                gd_init_sd, 
                                                                                gd_lr, 
                                                                                gd_num_iters, 
                                                                                gamma, 
                                                                                ensemble_size,
                                                                                compute_hessian,
                                                                                lmc_burn_in,
                                                                                lmc_lr_init,
                                                                                lmc_lr_decaying )   # MAP
            l_model_jkp_ml_.append(                     model_jkp_ml_)
            l_model_jkp_map.append(                     model_jkp_map)
            l_ens_jkp_hes.append(                       ens_jkp_hes)
            l_ens_jkp_fim.append(                       ens_jkp_fim)
            l_ens_jkp_lmc.append(                       ens_jkp_lmc)
        v_NC_jkp_ml_ =                                  torch.zeros(N)
        v_NC_jkp_map =                                  torch.zeros(N)
        v_NC_jkp_hes =                                  torch.zeros(N)
        v_NC_jkp_fim =                                  torch.zeros(N)
        v_NC_jkp_lmc =                                  torch.zeros(N)
        v_NC_jkp_ml__giq =                              torch.zeros(N)
        v_NC_jkp_map_giq =                              torch.zeros(N)
        v_NC_jkp_hes_giq =                              torch.zeros(N)
        v_NC_jkp_fim_giq =                              torch.zeros(N)
        v_NC_jkp_lmc_giq =                              torch.zeros(N)
        X_pairs =                                       D_te.X.repeat_interleave(len(Y_prime),dim=0)
        y_pairs =                                       Y_prime.repeat(D_te.X.shape[0])
        N_pairs =                                       N_te*N_prime
        m_NC_jkp_ml__prs =                              torch.zeros((N_te,N_prime,N))
        m_NC_jkp_map_prs =                              torch.zeros((N_te,N_prime,N))
        m_NC_jkp_hes_prs =                              torch.zeros((N_te,N_prime,N))
        m_NC_jkp_fim_prs =                              torch.zeros((N_te,N_prime,N))
        m_NC_jkp_lmc_prs =                              torch.zeros((N_te,N_prime,N))
        m_NC_jkp_ml__prs_giq =                          torch.zeros((N_te,N_prime,N))
        m_NC_jkp_map_prs_giq =                          torch.zeros((N_te,N_prime,N))
        m_NC_jkp_hes_prs_giq =                          torch.zeros((N_te,N_prime,N))
        m_NC_jkp_fim_prs_giq =                          torch.zeros((N_te,N_prime,N))
        m_NC_jkp_lmc_prs_giq =                          torch.zeros((N_te,N_prime,N))
        for i in range(N):
            sweep_i_str =                               f" sample ={i}/{N - 1}"
            print(print_str + sim_iter_str + sweep_iter_str + ' jkp NC' + sweep_i_str)
            x_i =                                       D.X[i,:].view(1,-1)
            y_i =                                       D.y[i].view(-1)
            v_NC_jkp_ml_[i] =                           nonconformity_frq(      x_i,     y_i,     l_model_jkp_ml_[i]                    )
            v_NC_jkp_map[i] =                           nonconformity_frq(      x_i,     y_i,     l_model_jkp_map[i]                    )
            v_NC_jkp_hes[i] =                           nonconformity_bay(      x_i,     y_i,       model_jkp_ens,   l_ens_jkp_hes[i]   )
            v_NC_jkp_fim[i] =                           nonconformity_bay(      x_i,     y_i,       model_jkp_ens,   l_ens_jkp_fim[i]   )
            v_NC_jkp_lmc[i] =                           nonconformity_bay(      x_i,     y_i,       model_jkp_ens,   l_ens_jkp_lmc[i]   )
            v_NC_jkp_ml__giq[i] =                       nonconformity_frq_giq(  x_i,     y_i,     l_model_jkp_ml_[i]                    )
            v_NC_jkp_map_giq[i] =                       nonconformity_frq_giq(  x_i,     y_i,     l_model_jkp_map[i]                    )
            v_NC_jkp_hes_giq[i] =                       nonconformity_bay_giq(  x_i,     y_i,       model_jkp_ens,   l_ens_jkp_hes[i]   )
            v_NC_jkp_fim_giq[i] =                       nonconformity_bay_giq(  x_i,     y_i,       model_jkp_ens,   l_ens_jkp_fim[i]   )
            v_NC_jkp_lmc_giq[i] =                       nonconformity_bay_giq(  x_i,     y_i,       model_jkp_ens,   l_ens_jkp_lmc[i]   )
            m_NC_jkp_ml__prs[:,:,i] =                   nonconformity_frq(      X_pairs, y_pairs, l_model_jkp_ml_[i]                    ).view(N_te,N_prime)
            m_NC_jkp_map_prs[:,:,i] =                   nonconformity_frq(      X_pairs, y_pairs, l_model_jkp_map[i]                    ).view(N_te,N_prime)
            m_NC_jkp_hes_prs[:,:,i] =                   nonconformity_bay(      X_pairs, y_pairs,   model_jkp_ens,   l_ens_jkp_hes[i]   ).view(N_te,N_prime)
            m_NC_jkp_fim_prs[:,:,i] =                   nonconformity_bay(      X_pairs, y_pairs,   model_jkp_ens,   l_ens_jkp_fim[i]   ).view(N_te,N_prime)
            m_NC_jkp_lmc_prs[:,:,i] =                   nonconformity_bay(      X_pairs, y_pairs,   model_jkp_ens,   l_ens_jkp_lmc[i]   ).view(N_te,N_prime)
            m_NC_jkp_ml__prs_giq[:,:,i] =               nonconformity_frq_giq(  X_pairs, y_pairs, l_model_jkp_ml_[i]                    ).view(N_te,N_prime)
            m_NC_jkp_map_prs_giq[:,:,i] =               nonconformity_frq_giq(  X_pairs, y_pairs, l_model_jkp_map[i]                    ).view(N_te,N_prime)
            m_NC_jkp_hes_prs_giq[:,:,i] =               nonconformity_bay_giq(  X_pairs, y_pairs,   model_jkp_ens,   l_ens_jkp_hes[i]   ).view(N_te,N_prime)
            m_NC_jkp_fim_prs_giq[:,:,i] =               nonconformity_bay_giq(  X_pairs, y_pairs,   model_jkp_ens,   l_ens_jkp_fim[i]   ).view(N_te,N_prime)
            m_NC_jkp_lmc_prs_giq[:,:,i] =               nonconformity_bay_giq(  X_pairs, y_pairs,   model_jkp_ens,   l_ens_jkp_lmc[i]   ).view(N_te,N_prime)


        s_jkp    ['covrg'][i_N,i_ml_,i_s],s_jkp    ['ineff'][i_N,i_ml_,i_s],s_jkp    ['covrg_labels'][i_N,i_ml_,i_s,:], s_jkp    ['ineff_labels'][i_N,i_ml_,i_s,:] =   jkp_covrg_and_ineff(m_NC_jkp_ml__prs,     v_NC_jkp_ml_,     D_te.y, alpha_cvb)
        s_jkp    ['covrg'][i_N,i_map,i_s],s_jkp    ['ineff'][i_N,i_map,i_s],s_jkp    ['covrg_labels'][i_N,i_map,i_s,:], s_jkp    ['ineff_labels'][i_N,i_map,i_s,:] =   jkp_covrg_and_ineff(m_NC_jkp_map_prs,     v_NC_jkp_map,     D_te.y, alpha_cvb)
        s_jkp    ['covrg'][i_N,i_hes,i_s],s_jkp    ['ineff'][i_N,i_hes,i_s],s_jkp    ['covrg_labels'][i_N,i_hes,i_s,:], s_jkp    ['ineff_labels'][i_N,i_hes,i_s,:] =   jkp_covrg_and_ineff(m_NC_jkp_hes_prs,     v_NC_jkp_hes,     D_te.y, alpha_cvb)
        s_jkp    ['covrg'][i_N,i_fim,i_s],s_jkp    ['ineff'][i_N,i_fim,i_s],s_jkp    ['covrg_labels'][i_N,i_fim,i_s,:], s_jkp    ['ineff_labels'][i_N,i_fim,i_s,:] =   jkp_covrg_and_ineff(m_NC_jkp_fim_prs,     v_NC_jkp_fim,     D_te.y, alpha_cvb)
        s_jkp    ['covrg'][i_N,i_lmc,i_s],s_jkp    ['ineff'][i_N,i_lmc,i_s],s_jkp    ['covrg_labels'][i_N,i_lmc,i_s,:], s_jkp    ['ineff_labels'][i_N,i_lmc,i_s,:] =   jkp_covrg_and_ineff(m_NC_jkp_lmc_prs,     v_NC_jkp_lmc,     D_te.y, alpha_cvb)
        s_jkp_giq['covrg'][i_N,i_ml_,i_s],s_jkp_giq['ineff'][i_N,i_ml_,i_s],s_jkp_giq['covrg_labels'][i_N,i_ml_,i_s,:], s_jkp_giq['ineff_labels'][i_N,i_ml_,i_s,:] =   jkp_covrg_and_ineff(m_NC_jkp_ml__prs_giq, v_NC_jkp_ml__giq, D_te.y, alpha_cvb)
        s_jkp_giq['covrg'][i_N,i_map,i_s],s_jkp_giq['ineff'][i_N,i_map,i_s],s_jkp_giq['covrg_labels'][i_N,i_map,i_s,:], s_jkp_giq['ineff_labels'][i_N,i_map,i_s,:] =   jkp_covrg_and_ineff(m_NC_jkp_map_prs_giq, v_NC_jkp_map_giq, D_te.y, alpha_cvb)
        s_jkp_giq['covrg'][i_N,i_hes,i_s],s_jkp_giq['ineff'][i_N,i_hes,i_s],s_jkp_giq['covrg_labels'][i_N,i_hes,i_s,:], s_jkp_giq['ineff_labels'][i_N,i_hes,i_s,:] =   jkp_covrg_and_ineff(m_NC_jkp_hes_prs_giq, v_NC_jkp_hes_giq, D_te.y, alpha_cvb)
        s_jkp_giq['covrg'][i_N,i_fim,i_s],s_jkp_giq['ineff'][i_N,i_fim,i_s],s_jkp_giq['covrg_labels'][i_N,i_fim,i_s,:], s_jkp_giq['ineff_labels'][i_N,i_fim,i_s,:] =   jkp_covrg_and_ineff(m_NC_jkp_fim_prs_giq, v_NC_jkp_fim_giq, D_te.y, alpha_cvb)
        s_jkp_giq['covrg'][i_N,i_lmc,i_s],s_jkp_giq['ineff'][i_N,i_lmc,i_s],s_jkp_giq['covrg_labels'][i_N,i_lmc,i_s,:], s_jkp_giq['ineff_labels'][i_N,i_lmc,i_s,:] =   jkp_covrg_and_ineff(m_NC_jkp_lmc_prs_giq, v_NC_jkp_lmc_giq, D_te.y, alpha_cvb)

        ## kfp = jackknife+
        print(print_str + sim_iter_str + sweep_iter_str + ' kfp fitting')
        NoverK =                        round(N / K)
        assert(                         NoverK * K == N )
        l_model_kfp_ml_ =               []
        l_model_kfp_map =               []
        l_ens_kfp_hes =                 []
        l_ens_kfp_fim =                 []
        l_ens_kfp_lmc =                 []
        for k in range(K):
            sweep_k_str =                               f" sample ={k}/{K - 1}"
            print(print_str + sim_iter_str + sweep_iter_str + ' kfp fitting' + sweep_k_str)
            model_kfp_ml_ =                             deepcopy(               net_init) # important this has the same init vector like all others
            model_kfp_map =                             deepcopy(               net_init) # "
            model_kfp_ens =                             deepcopy(               net_init) # to be used in the Bayes, changes every call of nonconformity_bay
            D_lfo =                                     D.leave_fold_out(k, K)
            m_loss_tr_kfp_ml_[i_s,i_N,k,:],\
                m_loss_te_kfp_ml_[i_s,i_N,k,:] =        fitting_erm_ml__gd(     model_kfp_ml_, D_lfo, D_te, gd_init_sd, gd_lr, gd_num_iters                          )   # ML
            m_loss_tr_kfp_map[i_s,i_N,k,:],\
                m_loss_te_kfp_map[i_s,i_N,k,:],\
                m_loss_tr_kfp_lmc[i_s,i_N,k,:],\
                m_loss_te_kfp_lmc[i_s,i_N,k,:],\
                ens_kfp_hes,\
                ens_kfp_fim,\
                ens_kfp_lmc =                           fitting_erm_map_gd(     model_kfp_map,
                                                                                D_lfo, 
                                                                                D_te, 
                                                                                gd_init_sd, 
                                                                                gd_lr, 
                                                                                gd_num_iters, 
                                                                                gamma, 
                                                                                ensemble_size,
                                                                                compute_hessian,
                                                                                lmc_burn_in,
                                                                                lmc_lr_init,
                                                                                lmc_lr_decaying )   # MAP
            l_model_kfp_ml_.append(                     model_kfp_ml_)
            l_model_kfp_map.append(                     model_kfp_map)
            l_ens_kfp_hes.append(                       ens_kfp_hes)
            l_ens_kfp_fim.append(                       ens_kfp_fim)
            l_ens_kfp_lmc.append(                       ens_kfp_lmc)
        v_NC_kfp_ml_ =                                  torch.zeros(N)
        v_NC_kfp_map =                                  torch.zeros(N)
        v_NC_kfp_hes =                                  torch.zeros(N)
        v_NC_kfp_fim =                                  torch.zeros(N)
        v_NC_kfp_lmc =                                  torch.zeros(N)
        v_NC_kfp_ml__giq =                              torch.zeros(N)
        v_NC_kfp_map_giq =                              torch.zeros(N)
        v_NC_kfp_hes_giq =                              torch.zeros(N)
        v_NC_kfp_fim_giq =                              torch.zeros(N)
        v_NC_kfp_lmc_giq =                              torch.zeros(N)
        X_pairs =                                       D_te.X.repeat_interleave(len(Y_prime),dim=0)
        y_pairs =                                       Y_prime.repeat(D_te.X.shape[0])
        N_pairs =                                       N_te*N_prime
        m_NC_kfp_ml__prs =                              torch.zeros((N_te,N_prime,N))
        m_NC_kfp_map_prs =                              torch.zeros((N_te,N_prime,N))
        m_NC_kfp_hes_prs =                              torch.zeros((N_te,N_prime,N))
        m_NC_kfp_fim_prs =                              torch.zeros((N_te,N_prime,N))
        m_NC_kfp_lmc_prs =                              torch.zeros((N_te,N_prime,N))
        m_NC_kfp_ml__prs_giq =                          torch.zeros((N_te,N_prime,N))
        m_NC_kfp_map_prs_giq =                          torch.zeros((N_te,N_prime,N))
        m_NC_kfp_hes_prs_giq =                          torch.zeros((N_te,N_prime,N))
        m_NC_kfp_fim_prs_giq =                          torch.zeros((N_te,N_prime,N))
        m_NC_kfp_lmc_prs_giq =                          torch.zeros((N_te,N_prime,N))
        v_k =                                           torch.arange(K).repeat_interleave(NoverK)   # mapping between index i to its fold k
        for i in range(N):
            sweep_i_str =                               f" sample ={i}/{N - 1}"
            print(print_str + sim_iter_str + sweep_iter_str + ' kfp NC' + sweep_i_str)
            x_i =                                       D.X[i,:].view(1,-1)
            y_i =                                       D.y[i].view(-1)
            v_NC_kfp_ml_[i] =                           nonconformity_frq(      x_i,     y_i,     l_model_kfp_ml_[v_k[i]]                        )
            v_NC_kfp_map[i] =                           nonconformity_frq(      x_i,     y_i,     l_model_kfp_map[v_k[i]]                        )
            v_NC_kfp_hes[i] =                           nonconformity_bay(      x_i,     y_i,       model_kfp_ens,       l_ens_kfp_hes[v_k[i]]   )
            v_NC_kfp_fim[i] =                           nonconformity_bay(      x_i,     y_i,       model_kfp_ens,       l_ens_kfp_fim[v_k[i]]   )
            v_NC_kfp_lmc[i] =                           nonconformity_bay(      x_i,     y_i,       model_kfp_ens,       l_ens_kfp_lmc[v_k[i]]   )
            v_NC_kfp_ml__giq[i] =                       nonconformity_frq_giq(  x_i,     y_i,     l_model_kfp_ml_[v_k[i]]                        )
            v_NC_kfp_map_giq[i] =                       nonconformity_frq_giq(  x_i,     y_i,     l_model_kfp_map[v_k[i]]                        )
            v_NC_kfp_hes_giq[i] =                       nonconformity_bay_giq(  x_i,     y_i,       model_kfp_ens,       l_ens_kfp_hes[v_k[i]]   )
            v_NC_kfp_fim_giq[i] =                       nonconformity_bay_giq(  x_i,     y_i,       model_kfp_ens,       l_ens_kfp_fim[v_k[i]]   )
            v_NC_kfp_lmc_giq[i] =                       nonconformity_bay_giq(  x_i,     y_i,       model_kfp_ens,       l_ens_kfp_lmc[v_k[i]]   )
            m_NC_kfp_ml__prs[:,:,i] =                   nonconformity_frq(      X_pairs, y_pairs, l_model_kfp_ml_[v_k[i]]                        ).view(N_te,N_prime)
            m_NC_kfp_map_prs[:,:,i] =                   nonconformity_frq(      X_pairs, y_pairs, l_model_kfp_map[v_k[i]]                        ).view(N_te,N_prime)
            m_NC_kfp_hes_prs[:,:,i] =                   nonconformity_bay(      X_pairs, y_pairs,   model_kfp_ens,       l_ens_kfp_hes[v_k[i]]   ).view(N_te,N_prime)
            m_NC_kfp_fim_prs[:,:,i] =                   nonconformity_bay(      X_pairs, y_pairs,   model_kfp_ens,       l_ens_kfp_fim[v_k[i]]   ).view(N_te,N_prime)
            m_NC_kfp_lmc_prs[:,:,i] =                   nonconformity_bay(      X_pairs, y_pairs,   model_kfp_ens,       l_ens_kfp_lmc[v_k[i]]   ).view(N_te,N_prime)
            m_NC_kfp_ml__prs_giq[:,:,i] =               nonconformity_frq_giq(  X_pairs, y_pairs, l_model_kfp_ml_[v_k[i]]                        ).view(N_te,N_prime)
            m_NC_kfp_map_prs_giq[:,:,i] =               nonconformity_frq_giq(  X_pairs, y_pairs, l_model_kfp_map[v_k[i]]                        ).view(N_te,N_prime)
            m_NC_kfp_hes_prs_giq[:,:,i] =               nonconformity_bay_giq(  X_pairs, y_pairs,   model_kfp_ens,       l_ens_kfp_hes[v_k[i]]   ).view(N_te,N_prime)
            m_NC_kfp_fim_prs_giq[:,:,i] =               nonconformity_bay_giq(  X_pairs, y_pairs,   model_kfp_ens,       l_ens_kfp_fim[v_k[i]]   ).view(N_te,N_prime)
            m_NC_kfp_lmc_prs_giq[:,:,i] =               nonconformity_bay_giq(  X_pairs, y_pairs,   model_kfp_ens,       l_ens_kfp_lmc[v_k[i]]   ).view(N_te,N_prime)
        s_kfp    ['covrg'][i_N,i_ml_,i_s],s_kfp    ['ineff'][i_N,i_ml_,i_s],s_kfp    ['covrg_labels'][i_N,i_ml_,i_s,:],s_kfp    ['ineff_labels'][i_N,i_ml_,i_s,:] =   kfp_covrg_and_ineff(m_NC_kfp_ml__prs,     v_NC_kfp_ml_,     D_te.y, alpha_cvb)
        s_kfp    ['covrg'][i_N,i_map,i_s],s_kfp    ['ineff'][i_N,i_map,i_s],s_kfp    ['covrg_labels'][i_N,i_map,i_s,:],s_kfp    ['ineff_labels'][i_N,i_map,i_s,:] =   kfp_covrg_and_ineff(m_NC_kfp_map_prs,     v_NC_kfp_map,     D_te.y, alpha_cvb)
        s_kfp    ['covrg'][i_N,i_hes,i_s],s_kfp    ['ineff'][i_N,i_hes,i_s],s_kfp    ['covrg_labels'][i_N,i_hes,i_s,:],s_kfp    ['ineff_labels'][i_N,i_hes,i_s,:] =   kfp_covrg_and_ineff(m_NC_kfp_hes_prs,     v_NC_kfp_hes,     D_te.y, alpha_cvb)
        s_kfp    ['covrg'][i_N,i_fim,i_s],s_kfp    ['ineff'][i_N,i_fim,i_s],s_kfp    ['covrg_labels'][i_N,i_fim,i_s,:],s_kfp    ['ineff_labels'][i_N,i_fim,i_s,:] =   kfp_covrg_and_ineff(m_NC_kfp_fim_prs,     v_NC_kfp_fim,     D_te.y, alpha_cvb)
        s_kfp    ['covrg'][i_N,i_lmc,i_s],s_kfp    ['ineff'][i_N,i_lmc,i_s],s_kfp    ['covrg_labels'][i_N,i_lmc,i_s,:],s_kfp    ['ineff_labels'][i_N,i_lmc,i_s,:] =   kfp_covrg_and_ineff(m_NC_kfp_lmc_prs,     v_NC_kfp_lmc,     D_te.y, alpha_cvb)
        s_kfp_giq['covrg'][i_N,i_ml_,i_s],s_kfp_giq['ineff'][i_N,i_ml_,i_s],s_kfp_giq['covrg_labels'][i_N,i_ml_,i_s,:],s_kfp_giq['ineff_labels'][i_N,i_ml_,i_s,:] =   kfp_covrg_and_ineff(m_NC_kfp_ml__prs_giq, v_NC_kfp_ml__giq, D_te.y, alpha_cvb)
        s_kfp_giq['covrg'][i_N,i_map,i_s],s_kfp_giq['ineff'][i_N,i_map,i_s],s_kfp_giq['covrg_labels'][i_N,i_map,i_s,:],s_kfp_giq['ineff_labels'][i_N,i_map,i_s,:] =   kfp_covrg_and_ineff(m_NC_kfp_map_prs_giq, v_NC_kfp_map_giq, D_te.y, alpha_cvb)
        s_kfp_giq['covrg'][i_N,i_hes,i_s],s_kfp_giq['ineff'][i_N,i_hes,i_s],s_kfp_giq['covrg_labels'][i_N,i_hes,i_s,:],s_kfp_giq['ineff_labels'][i_N,i_hes,i_s,:] =   kfp_covrg_and_ineff(m_NC_kfp_hes_prs_giq, v_NC_kfp_hes_giq, D_te.y, alpha_cvb)
        s_kfp_giq['covrg'][i_N,i_fim,i_s],s_kfp_giq['ineff'][i_N,i_fim,i_s],s_kfp_giq['covrg_labels'][i_N,i_fim,i_s,:],s_kfp_giq['ineff_labels'][i_N,i_fim,i_s,:] =   kfp_covrg_and_ineff(m_NC_kfp_fim_prs_giq, v_NC_kfp_fim_giq, D_te.y, alpha_cvb)
        s_kfp_giq['covrg'][i_N,i_lmc,i_s],s_kfp_giq['ineff'][i_N,i_lmc,i_s],s_kfp_giq['covrg_labels'][i_N,i_lmc,i_s,:],s_kfp_giq['ineff_labels'][i_N,i_lmc,i_s,:] =   kfp_covrg_and_ineff(m_NC_kfp_lmc_prs_giq, v_NC_kfp_lmc_giq, D_te.y, alpha_cvb)

    file_str = f'run0.mat'
    savemat(path_of_run + file_str, {   "dSetting":                         dSetting,
                                        "v_N":                              v_N,
                                        "gamma":                            gamma,
                                        "i_s":                              i_s,
                                        "v_alg_str":                        v_alg_str,
                                        "i_ml_":                            i_ml_,
                                        "i_map":                            i_map,
                                        "i_hes":                            i_hes,
                                        "i_fim":                            i_fim,
                                        "i_lmc":                            i_lmc,
                                        "alpha":                            alpha,
                                        "alpha_cvb":                        alpha_cvb,
                                        "K":                                K,
                                        "tPercentiles":                     tPercentiles,
                                        "gd_num_iters":                     gd_num_iters,
                                        "gd_lr":                            gd_lr,
                                        "lmc_burn_in":                      lmc_burn_in,
                                        "lmc_lr_init":                      lmc_lr_init,
                                        "lmc_lr_decaying":                  lmc_lr_decaying,
                                        "lmc_num_iters":                    lmc_num_iters,
                                        "ensemble_size":                    ensemble_size,
                                        "compute_hessian":                  compute_hessian,
                                        "s_sft":                            s_sft,
                                        "s_vb_":                            s_vb_,
                                        "s_jkp":                            s_jkp,
                                        "s_kfp":                            s_kfp,
                                        "s_vb__giq":                        s_vb__giq,
                                        "s_jkp_giq":                        s_jkp_giq,
                                        "s_kfp_giq":                        s_kfp_giq,
                                        "m_loss_tr_sft_ml_":                m_loss_tr_sft_ml_.numpy(),
                                        "m_loss_te_sft_ml_":                m_loss_te_sft_ml_.numpy(),
                                        "m_loss_tr_sft_map":                m_loss_tr_sft_map.numpy(),
                                        "m_loss_te_sft_map":                m_loss_te_sft_map.numpy(),
                                        "m_loss_tr_vb__ml_":                m_loss_tr_vb__ml_.numpy(),
                                        "m_loss_te_vb__ml_":                m_loss_te_vb__ml_.numpy(),
                                        "m_loss_tr_vb__map":                m_loss_tr_vb__map.numpy(),
                                        "m_loss_te_vb__map":                m_loss_te_vb__map.numpy(),
                                        "m_loss_tr_jkp_ml_":                m_loss_tr_jkp_ml_.numpy(),
                                        "m_loss_te_jkp_ml_":                m_loss_te_jkp_ml_.numpy(),
                                        "m_loss_tr_jkp_map":                m_loss_tr_jkp_map.numpy(),
                                        "m_loss_te_jkp_map":                m_loss_te_jkp_map.numpy(),
                                        "m_loss_tr_kfp_ml_":                m_loss_tr_kfp_ml_.numpy(),
                                        "m_loss_te_kfp_ml_":                m_loss_te_kfp_ml_.numpy(),
                                        "m_loss_tr_kfp_map":                m_loss_tr_kfp_map.numpy(),
                                        "m_loss_te_kfp_map":                m_loss_te_kfp_map.numpy(),
                                        "m_loss_tr_sft_lmc":                m_loss_tr_sft_lmc.numpy(),
                                        "m_loss_te_sft_lmc":                m_loss_te_sft_lmc.numpy(),
                                        "m_loss_tr_vb__lmc":                m_loss_tr_vb__lmc.numpy(),
                                        "m_loss_te_vb__lmc":                m_loss_te_vb__lmc.numpy(),
                                        "m_loss_tr_jkp_lmc":                m_loss_tr_jkp_lmc.numpy(),
                                        "m_loss_te_jkp_lmc":                m_loss_te_jkp_lmc.numpy(),
                                        "m_loss_tr_kfp_lmc":                m_loss_tr_kfp_lmc.numpy(),
                                        "m_loss_te_kfp_lmc":                m_loss_te_kfp_lmc.numpy(),
                                    } )
    print('Saving to file '+file_str)

itr_file.close()
print('run0.py done. Total time:')
print(datetime.datetime.now() - startTime)

