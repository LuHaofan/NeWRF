import scipy
import numpy as np
import pandas as pd
import torch
import os

class DatasetParser:
    def __init__(self, dataset_file) -> None:
        self.dataset_file = dataset_file
        self.dataset = self._read_mat_dataset()

    def _read_mat_dataset(self):
        dataset = scipy.io.loadmat(self.dataset_file)
        try:
            dataset = dataset['dataset']
            cols = dataset[:,1].dtype.names
        except:
            dataset = dataset['subdataset']
            cols = dataset[:,0].dtype.names
        d = {n : dataset[n] for n in cols}

        for k in d.keys():
            d[k] = np.concatenate([d[k][:,i] for i in range(d[k].shape[1])])

        df = pd.DataFrame(d)

        # Unbox the data from nested array
        for col in df.columns:
            if col in ['TxPos', 'RxPos', 'CSI', 'PathLengths', 'PathGains']:
                df[col] = df[col].apply(self.strip_single_column)
            elif col in ['TxID', 'RxID', 'Frequency']:
                df[col] = df[col].apply(self.strip_single_value)
            elif col == 'AoA':
                df[col] = df[col].apply(self.strip_aoa)
            elif col in ['LastXPts', 'LastXptChans', 'LastXptViewdirs', 'LastInteractionPts']:
                df[col] = df[col].apply(self.strip_last_interaction_point)

        # Remove the NAN rows
        return df[df['TxPos'].notna()]
    
    def save_to_pkl(self, surfix=""):
        fname = self.dataset_file.split('/')[-1]
        self.dataset.to_pickle(os.path.join("./pkl/", fname[:-4]+surfix+".pkl"))  
    
    def strip_single_column(self, val):
        if val.ndim > 1:
            return val.flatten()
        else:
            return np.nan

    def strip_single_value(self, val):
        n_ele = np.prod(val.shape)
        if n_ele == 1:
            return val[0,0]
        else:
            return np.nan
    
    def strip_aoa(self, val):
        if val.ndim > 1:
            return val.T
        else:
            return np.nan
        
    def strip_last_interaction_point(self, val):
        if val.ndim > 1:
            return val
        else:
            return np.nan


class DatasetLoader:
    def __init__(self, dataset_file) -> None:
        self.dataset_file = dataset_file
        self.dataset = self._read_pkl_dataset()
        self.dataset = self.dataset.dropna()
        self.trainset = []
        self.valset = []

    def _read_pkl_dataset(self):
        return pd.read_pickle(self.dataset_file)  

    def get_cfr(self, tx_id, rx_id, fc_lst = None, normalize = False):
        '''
        Return the Channel Frequency Response between the pair (tx_id, rx_id)
        at the specified center frequency. If the center frequency fc is not specified,
        by default return the CFR for all frequencies.
        '''
        pair = self.dataset[(self.dataset['TxID'] == tx_id) & (self.dataset['RxID'] == rx_id)]
        cfr_lst = []
        if fc_lst is not None:
            for fc in fc_lst:
                cfr_lst.append(pair[pair['Frequency'] == fc]['CSI'].to_numpy()[0])
        else:
            cfr_lst = pair.sort_values(by = 'Frequency')['CSI'].to_numpy()
        cfr = np.stack(cfr_lst)
        if normalize:
            real_cfr = ((np.real(cfr)-self.min_real)/(self.max_real - self.min_real)) * 2 - 1
            imag_cfr = ((np.imag(cfr)-self.min_imag)/(self.max_imag - self.min_imag)) * 2 - 1
            cfr = real_cfr + 1j * imag_cfr
        return cfr
    
    def get_cfr_batch(self, tx_id, rx_ids, fc_lst = None, normalize = False):
        cfr_lst = []
        for i in rx_ids:
            cfr_lst.append(self.get_cfr(tx_id, i, fc_lst, normalize).flatten())
        return cfr_lst
    
    def fspl(self, d, f):
        return 20*np.log10(d) + 20*np.log10(f) + 20*np.log10(4*np.pi/299792458)
    
    def get_cfr_per_ray_batch(self, tx_id, rx_ids, cfg):
        cfr_lst = []
        if "RxChanPerRay" in self.dataset.columns:
            for sta_id in rx_ids:
                row = self.get_pair(tx_id, sta_id)
                cfr_lst.append(row["RxChanPerRay"].to_numpy()[0])
        else:
            print("Warning: calculating the ground truth in run time, this might be slow!")
            for sta_id in rx_ids:
                row = self.get_pair(tx_id, sta_id)
                fc = row['Frequency'].to_numpy()[0]*1e9
                fsc = fc + np.array(cfg.ofdm.ActiveFrequencyIndices)*float(cfg.ofdm.SubcarrierSpacing)
                lastXpts = row['LastXPts'].to_numpy()[0]
                lastXptChans = row['LastXptChans'].to_numpy()[0]
                rx_loc = row['RxPos'].to_numpy()[0]
                rx_loc_full = np.broadcast_to(np.expand_dims(rx_loc, axis = -1), lastXpts.shape)
                lastXptDists = np.linalg.norm(lastXpts-rx_loc_full, ord = 2, axis = 0)
                n_paths = lastXpts.shape[-1]
                path_coeffs = []
                for k in range(n_paths):
                    d = lastXptDists[k]
                    pathloss = self.fspl(d, fsc)
                    phs_shift = -2*np.pi*fsc*d/299792458
                    rx_chan = lastXptChans[:,k]*(10**(-pathloss/20))*np.exp(1j*phs_shift)
                    path_coeffs.append(rx_chan)
                path_coeffs = np.stack(path_coeffs)
                cfr_lst.append(path_coeffs)
        return cfr_lst
    
    def get_all_center_freqs(self):
        return np.array(sorted(self.dataset['Frequency'].unique()))
    
    def get_channel_freqs(self, n_chs):
        return np.array(sorted(self.dataset['Frequency'].unique())[:n_chs])
        
    def get_loc(self, dev_type, id, normalize = False):
        if normalize:
            if dev_type == "AP":
                dev_entries = self.dataset[self.dataset['TxID'] == id]['NormTxPos']
            elif dev_type == "STA":
                dev_entries = self.dataset[self.dataset['RxID'] == id]['NormRxPos']
            
        else:
            if dev_type == "AP":
                dev_entries = self.dataset[self.dataset['TxID'] == id]['TxPos']
            elif dev_type == "STA":
                dev_entries = self.dataset[self.dataset['RxID'] == id]['RxPos']
        return dev_entries[dev_entries.index[0]]
    
    def get_loc_batch(self, dev_type, ids, normalize = False):
        loc_lst = []
        for i in ids:
            loc_lst.append(self.get_loc(dev_type, i, normalize))
        return loc_lst
    
    def get_aoa(self, id):
        lastXpts = self.get_last_itx_pts(id)
        center = self.get_loc('STA', id)
        aoa = lastXpts-center
        return aoa/np.linalg.norm(aoa,axis=1)[:, None]
    
    def get_aoa_batch(self, ids):
        aoa_lst = []
        for i in ids:
            aoa_lst.append(self.get_aoa(i))
        return aoa_lst
    
    def get_last_itx_pts(self, id):
        dev_entries = self.dataset[self.dataset['RxID']==id]['LastXPts']
        return dev_entries[dev_entries.index[0]].T
    
    def get_station_ids(self):
        return self.dataset['RxID'].unique()
    
    def get_ap_ids(self):
        return self.dataset['TxID'].unique()
    
    def get_case(self, ap_id, sta_id, device, normalize_cfr = True, normalize_loc = False):
        # shape: [N_channels, N_subcarriers]
        cfr = self.get_cfr(ap_id, sta_id)
        cfr = torch.tensor(cfr, dtype=torch.complex64).to(device)
        if normalize_cfr:
            # cfr = (cfr-self.csi_norm_transform[0])/self.csi_norm_transform[1]
            real_cfr = ((np.real(cfr)-self.min_real)/(self.max_real - self.min_real)) * 2 - 1
            imag_cfr = ((np.imag(cfr)-self.min_imag)/(self.max_imag - self.min_imag)) * 2 - 1
            cfr = real_cfr + 1j * imag_cfr

        sta_loc = torch.tensor(self.get_loc(dev_type='STA', id=sta_id, normalize=normalize_loc), dtype=torch.float32)
        return cfr, sta_loc
    
    def get_pair(self, ap_id, sta_id):
        pair = self.dataset[(self.dataset['TxID'] == ap_id) & (self.dataset['RxID'] == sta_id)]
        return pair
        
    def split_train_val(self, ratio = 0.8, fixed_val = True):
        all_rx = self.dataset.RxID.unique()
        if fixed_val:
            n_valid = int(len(all_rx)*0.2)
            n_train = int(len(all_rx)*ratio)
            self.valset = all_rx[int(-1*n_valid):]
            self.trainset = all_rx[:n_train]
        else:
            n_train = int(len(all_rx)*ratio)
            n_valid = len(all_rx)-n_train
            if n_valid < 1:
                n_valid = 1
                n_train = len(all_rx)-n_valid
            self.valset = all_rx[int(-1*n_valid):]
            self.trainset = all_rx[:int(-1*n_valid)]

    def _normalize_pos(self, pos, dx, dy, dz):
        assert(pos.shape == (3,))
        return [pos[0]*2/dx, pos[1]*2/dy, pos[2]*2/dz]
    
    def _normalize_freq(self, freq, mean_f, std_f):
        return (freq-mean_f)/std_f

    def normalize_frequency(self):
        mean_freq, std_freq = self.dataset.Frequency.mean(), self.dataset.Frequency.std()
        self.dataset['NormFrequency'] = self.dataset['Frequency'].apply(self._normalize_freq, args = (mean_freq, std_freq))

    def normalize_rx_pos(self, cfg):
        env_dim = cfg.dataset.env_dim
        dx = env_dim[1]-env_dim[0]
        dy = env_dim[3]-env_dim[2]
        dz = env_dim[5]-env_dim[4]
        self.dataset['NormRxPos'] = self.dataset['RxPos'].apply(self._normalize_pos, args=(dx, dy, dz))
