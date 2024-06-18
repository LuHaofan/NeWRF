import numpy as np
import pandas as pd
from typing import List

class DatasetLoader:
    def __init__(self, dataset_file) -> None:
        self.dataset = pd.read_pickle(dataset_file)
        self.dataset = self.dataset.dropna()
        self.trainset = []
        self.valset = []

    def get_pair(self, tx_id, rx_id):
        pair = self.dataset[(self.dataset['TxID'] == tx_id) & (self.dataset['RxID'] == rx_id)]
        return pair

    def get_cfr(self, tx_id, rx_id):
        pair = self.get_pair(tx_id, rx_id)
        return pair['CSI'].to_numpy()[0].astype(np.complex64)
    
    def get_cfr_batch(self, tx_id, rx_ids):
        cfr_lst = []
        for rx_id in rx_ids:
            cfr_lst.append(self.get_cfr(tx_id, rx_id).flatten())
        return np.array(cfr_lst)
    
    def get_freq(self):
        return self.dataset['Frequency'].unique()[0].astype(np.float32) 
    
    def get_loc(self, dev_type, id):
        if dev_type == "AP":
            dev_entries = self.dataset[self.dataset['TxID'] == id]['TxPos']
        elif dev_type == "STA":
            dev_entries = self.dataset[self.dataset['RxID'] == id]['RxPos']
        return dev_entries[dev_entries.index[0]]
    
    def get_loc_batch(self, dev_type, ids):
        loc_lst = []
        for i in ids:
            loc_lst.append(self.get_loc(dev_type, i))
        return np.array(loc_lst).astype(np.float32)
    
    def get_aoa(self, id):
        lastXpts = self.get_last_itx_pts(id)
        center = self.get_loc('STA', id)
        aoa = lastXpts-center
        return aoa/np.linalg.norm(aoa,axis=1)[:, None].astype(np.float32)
    
    def get_aoa_batch(self, ids) -> List[np.ndarray]:
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
        
    def split_train_val(self, ratio = 0.8, fixed_val = True):
        '''
        Split the dataset into training and validation sets.
        @param ratio: The ratio of the training set to the whole dataset.
        @param fixed_val: If True, the validation set will be fixed to the last 20% of the dataset.
        '''
        all_rx = self.get_station_ids()
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

if __name__ == "__main__":
    loader = DatasetLoader("./data/dataset_conference_ch1_rt_image_fc.pkl")
    loader.split_train_val()
    trainset = loader.trainset
    valset = loader.valset
    freq = loader.get_freq()
    loc_sta_1 = loader.get_loc('STA', 1)
    loc_batch_sta = loader.get_loc_batch('STA', [1, 2, 3])
    aoa_1 = loader.get_aoa(1)
    aoa_batch = loader.get_aoa_batch([1, 2, 3])
    last_itx_pts_1 = loader.get_last_itx_pts(1)
    station_ids = loader.get_station_ids()
    ap_ids = loader.get_ap_ids()
    cfr_1_1 = loader.get_cfr(1, 1)
    cfr_batch_1 = loader.get_cfr_batch(1, [1, 2, 3])

    print(f"trainset size: {trainset.shape}, type: {type(trainset)}")
    print(f"valset size: {valset.shape}, type: {type(valset)}")
    print(f"freq size: {freq.shape}, type: {type(freq)}, freq value: {freq}")
    print(f"loc_sta_1 size: {loc_sta_1.shape}, type: {type(loc_sta_1)}")
    print(f"loc_batch_sta size: {loc_batch_sta.shape}, type: {type(loc_batch_sta)}")
    print(f"aoa_1 size: {aoa_1.shape}, type: {type(aoa_1)}")
    print(f"aoa_batch size: {len(aoa_batch)}, type: {type(aoa_batch)}, aoa_batch[0] size: {aoa_batch[0].shape}")
    print(f"last_itx_pts_1 size: {last_itx_pts_1.shape}, type: {type(last_itx_pts_1)}")
    print(f"station_ids size: {station_ids.shape}, type: {type(station_ids)}")
    print(f"ap_ids size: {ap_ids.shape}, type: {type(ap_ids)}")
    print(f"cfr_1_1 size: {cfr_1_1.shape}, type: {type(cfr_1_1)}, dtype: {cfr_1_1.dtype}")
    print(f"cfr_batch_1 size: {cfr_batch_1.shape}, type: {type(cfr_batch_1)}, dtype: {cfr_batch_1.dtype}")