import os
import git
import h5py
import yaml
from torch.utils.tensorboard import SummaryWriter
from .data import MultiblockTensorDataset

def configure_tensorboard(proj_dir=None):
    tb_dir = os.path.join(proj_dir,'tensorboard')
    if os.path.exists(tb_dir):
        pass
    else:
        os.mkdir(tb_dir)
    writer = SummaryWriter(tb_dir)
    # TODO: add plotter function
    return writer

def get_test_ecog_dataset(device='cpu'):    
    file_dir = os.path.dirname(__file__)
    test_data_dir = os.path.join(os.path.dirname(file_dir),'tests','data')
    input_dataset = h5py.File(os.path.join(test_data_dir,'gw_250_nband3_test'),'r')
    target_dataset = h5py.File(os.path.join(test_data_dir,'gw_250_test'),'r')
    n_band = 3
    return MultiblockTensorDataset(
        target_data_record=target_dataset,
        band_data_record=input_dataset,
        n_band=n_band,
        device=device
    )

def get_repo_commit_hash():
    repo = git.Repo(
        path=os.path.dirname(__file__),
        search_parent_directories=True)
    return repo.head.object.hexsha

def write_repo_commit_hash(file_path):
    with open(file_path,'w') as fp:
        fp.write(get_repo_commit_hash())

def read_yaml(file_path):
    # for hyperparameter files
    with open(file_path,'r') as yf:
        yaml_data = yaml.load(yf)
    return yaml_data

def write_yaml(data,file_path):
    with open(file_path,'w') as yf:
        yaml.dump(data,yf)

if __name__ == "__main__":
    pass