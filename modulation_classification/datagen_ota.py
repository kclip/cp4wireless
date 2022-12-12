import h5py
import torch
import numpy as np
import sys

class Data_gen_OTA:
    def __init__(self, num_sampling_examples, snr_db_th=6, path_for_h5py=None):
        if path_for_h5py is None:
            if sys.platform.startswith('linux'):
                self.path_for_h5py = '/scratch/prj/nmes_simeone/radioml_2018/archive/GOLD_XYZ_OSC.0001_1024.hdf5'
            else:
                self.path_for_h5py = 'dataset/GOLD_XYZ_OSC.0001_1024.hdf5'
        else:
            pass
        self.snr_db_th =            snr_db_th
        self.f =                    h5py.File(self.path_for_h5py, "r") # total of 2555904 examples
        self.higher_snr_ind =       (self.f['Z'][:] >= self.snr_db_th).squeeze()
        self.X =                    self.f['X'][self.higher_snr_ind,:,:]
        self.Y =                    self.f['Y'][self.higher_snr_ind,:]
        self.Z =                    self.f['Z'][self.higher_snr_ind]
        self.f.close()

        self.num_sampling_examples = num_sampling_examples # this is N+N_te
    def gen(self, N, device='cpu'):
        # first generate random sequence with length total_num_examples
        rand_indices_for_sampling = torch.randperm(self.X.shape[0])[:self.num_sampling_examples]
        X = []
        Y = []
        for idx, curr_actual_index in enumerate(rand_indices_for_sampling):
            if idx%1000==0:
                print(f'generating {idx}/{self.num_sampling_examples} pairs with N={N}. last is pair number {curr_actual_index}/{len(self.Z)}')
            x = self.X[int(curr_actual_index)] # (1024, 2)
            x = np.transpose(x) # (2, 1024) (C, L)
            x = torch.from_numpy(x)
            y = int(np.nonzero(self.Y[int(curr_actual_index)])[0]) # one_hot vector to its nonzero index
            y = torch.tensor(y)
            X.append(x.unsqueeze(dim=0)) # (1,2,1024)
            Y.append(y.unsqueeze(dim=0)) # (1,1)
        X = torch.cat(X, dim=0).to(device) # (num_sampling_examples, 2, 1024)
        Y = torch.cat(Y, dim=0).to(device) # (num_sampling_examples, 1)
        Y = Y.unsqueeze(dim=1)
        tr_dataset = (X[:N], Y[:N]) # (N, 2, 1024), (N, 1) <-> (N, C, L), (N, 1)
        te_dataset = (X[N:], Y[N:]) # (N_te, 2, 1024), (N_te, 1)
        return tr_dataset, te_dataset