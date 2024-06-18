import numpy as np
import scipy
import os
import pandas as pd

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
            elif col in ['TxID', 'RxID', 'Frequency', 'LineOfSight']:
                df[col] = df[col].apply(self.strip_single_value)
            elif col in ['AoA', 'RxChanPerRay']:
                df[col] = df[col].apply(self.strip_aoa)
            elif col in ['LastXPts', 'LastXptChans', 'LastXptViewdirs', 'LastInteractionPts']:
                df[col] = df[col].apply(self.strip_last_interaction_point)

        # Remove the NAN rows
        return df[df['TxPos'].notna()]
    
    def save_to_pkl(self, data_dir, surfix=""):
        fname = self.dataset_file.split('/')[-1]
        self.dataset.to_pickle(os.path.join(data_dir, fname[:-4]+surfix+".pkl"))  
    
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

if __name__ == "__main__":
    data_dir = "./datasets/"
    fnames = ["dataset_bedroom.mat"]#os.listdir(data_dir)
    for f in fnames:
        parser = DatasetParser(os.path.join(data_dir, f))
        parser.save_to_pkl(data_dir, surfix="_971stas")