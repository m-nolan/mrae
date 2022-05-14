import os
import h5py
import yaml
import torch
from mrae import data, MRAE, objective

# # # RUN FROM PROJECT ROOT DIRECTORY # # #

# create run directory
current_dir = os.path.dirname(__file__)
run_dir = os.path.join(current_dir,'mrae_run_example')
if not os.path.exists(run_dir):
    os.makedirs(run_dir)

# load hyperparameter, save to run directory
hparam_read_file = os.path.join(current_dir,'hyperparameters','run_example.yaml')
with open(hparam_read_file,'r') as hpf:
    hparams = yaml.load(hpf)
hparam_write_file = os.path.join(run_dir,'hyperparameters.yaml')
with open(hparam_write_file,'w') as hpf:
    yaml.dump(hparams,hpf)

# create dataset and dataloaders (3-band dataset for 3-block model)
data_dir = os.path.join(current_dir,'tests','data')
target_dataset = h5py.File(os.path.join(data_dir,'gw_250_test'),'r')
input_dataset = h5py.File(os.path.join(data_dir,'gw_250_nband3_test'),'r')
n_block = 3
ecog_ds = data.MultiblockTensorDataset(
    target_dataset,
    input_dataset,
    n_band=n_block,
    device='cpu'
)
batch_size = 10
train_dl, valid_dl, test_dl = data.get_partition_dataloaders(ecog_ds,batch_size=batch_size)

# initialize model, objective from hyperparameter yaml file, dataset sizes
input_size = ecog_ds.target_data_record['ecog'].shape[-1]
mrae = MRAE.MRAE(
    input_size=input_size,
    encoder_size=hparams['model']['encoder_size'],
    decoder_size=hparams['model']['decoder_size'],
    num_blocks=hparams['model']['num_blocks'],
    dropout=hparams['model']['dropout'],
    num_layers_encoder=hparams['model']['num_layers_encoder'],
    bidirectional_encoder=hparams['model']['bidirectional_encoder'],
    device='cpu'
)

mrae_obj = objective.MRAEObjective(
    kl_div_scale_max=hparams['objective']['kl_div_scale_max'],
    l2_scale_max=hparams['objective']['l2_scale_max'],
    max_at_epoch=hparams['objective']['max_at_epoch']
)

# fit model to data
mrae.fit(
    train_dl=train_dl,
    valid_dl=valid_dl,
    objective=mrae_obj,
    save_dir=run_dir,
    min_epochs=hparams['run']['min_epochs'],
    max_epochs=hparams['run']['max_epochs'],
    n_search_epochs=hparams['run']['n_search_epochs'],
    overwrite=hparams['run']['overwrite']
)

# evaluate model performance on test dataset
