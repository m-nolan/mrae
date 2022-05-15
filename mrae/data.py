from collections import namedtuple
import torch
from torch.utils.data import Subset, DataLoader
from torch.utils.data.sampler import BatchSampler, SequentialSampler
import torch.nn as nn
import numpy as np

# output collection structure (named tuple, good for potential DDP)
mrae_output = namedtuple(
    'mrae_output',
    [
        'output',
        'block_output',
        'hidden',
        'decoder_ic',
        'decoder_ic_kl_div',
        'decoder_l2'
        ]
)

# dataloader classes
class MultiblockTensorDataset(torch.utils.data.Dataset):
    
    def __init__(self, target_data_record, band_data_record, n_band, device='cpu', dtype=torch.float32):
        if n_band > 1:
            assert target_data_record['ecog'].shape[-1] == band_data_record['ecog_band0'].shape[-1] # channel sizes match
        self.target_data_record = target_data_record
        self.band_data_record = band_data_record
        self.input_size = target_data_record['ecog'].shape[-1]
        print('aligning target, banded dataset samples...')
        self.set_shared_sample_idx()
        self.n_band = n_band
        self.device = device
        self.dtype = dtype

    def __getitem__(self, index):
        target = torch.tensor(
            self.target_data_record['ecog'][self.sample_idx[index,0],:,:],
            dtype=self.dtype
        ).to(self.device)
        if self.n_band > 1:
            input = torch.stack(
                [
                    torch.tensor(
                        self.band_data_record[f'ecog_band{b_idx}'][self.sample_idx[index,1],:,:],
                        dtype=self.dtype
                    ) for b_idx in range(self.n_band)
                ],
                dim = -1
            ).to(self.device)
        else:
            input = torch.stack(target,dim=-1)

        return input, target

    def __len__(self):
        return self.sample_idx.shape[0]

    def set_shared_sample_idx(self):
        print('computing intersection...')
        target_trial_loc = get_trial_location_view(self.target_data_record)
        band_trial_loc = get_trial_location_view(self.band_data_record)
        shared_trial_loc, target_data_loc_idx, band_data_loc_idx = np.intersect1d(
            target_trial_loc, band_trial_loc, assume_unique=True, return_indices=True
        )
        self.sample_idx = np.hstack(
            [
                target_data_loc_idx[:,None],
                band_data_loc_idx[:,None]
            ]
        )

def get_trial_location_view(record):
    dtype = record['dataset_idx'][()].dtype
    view_def = {
        'names': ['dataset_idx', 'trial_start_idx'],
        'formats': 2*[dtype]
    }
    return np.hstack(
        [
            record['dataset_idx'][()][:,None],
            record['trial_start_idx'][()][:,None]
        ]
    ).view(view_def)

def get_partition_dataloaders(ds, batch_size, partition={'train': 0.7, 'valid': 0.2, 'test': 0.1}):
    num_trials = len(ds)
    num_train_trials = round(num_trials * partition['train'])
    num_valid_trials = round(num_trials * partition['valid'])
    num_test_trials = round(num_trials * partition['test'])

    train_trial_idx = np.arange(num_trials)[:num_train_trials]
    valid_trial_idx = np.arange(num_trials)[num_train_trials:(num_train_trials+num_valid_trials)]
    test_trial_idx = np.arange(num_trials)[-num_test_trials:]

    train_dl = subset_batch_dataloader(ds, train_trial_idx, batch_size)
    valid_dl = subset_batch_dataloader(ds, valid_trial_idx, batch_size)
    test_dl = subset_batch_dataloader(ds, test_trial_idx, batch_size)

    return train_dl, valid_dl, test_dl

def subset_batch_dataloader(ds,subset_idx,batch_size):
    subset_ds = Subset(ds,subset_idx)
    subset_sampler = BatchSampler(
        SequentialSampler(subset_ds),
        batch_size=batch_size,
        drop_last=True
    )
    return DataLoader(subset_ds, sampler=subset_sampler)

if __name__ == "__main__":
    pass