import os
from mrae import data, MRAE, objective, utils

# # # RUN FROM PROJECT ROOT DIRECTORY # # #

BATCH_SIZE = 10
CURR_DIR = os.path.dirname(__file__)

def get_default_run_dir():
    run_dir = os.path.join(CURR_DIR,'mrae_run_example')
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    return run_dir

def mrae_model_from_hparams(input_size,hparams):
    return MRAE.MRAE(
        input_size=input_size,
        encoder_size=hparams['model']['encoder_size'],
        decoder_size=hparams['model']['decoder_size'],
        num_blocks=hparams['model']['num_blocks'],
        dropout=hparams['model']['dropout'],
        num_layers_encoder=hparams['model']['num_layers_encoder'],
        bidirectional_encoder=hparams['model']['bidirectional_encoder'],
        device='cpu'
    )

def mrae_obj_from_hparams(hparams):
    return objective.MRAEObjective(
        kl_div_scale_max=hparams['objective']['kl_div_scale_max'],
        l2_scale_max=hparams['objective']['l2_scale_max'],
        max_at_epoch=hparams['objective']['max_at_epoch']
    )

# evaluate model performance on test dataset

def main(args):
    # create run directory
    run_dir = get_default_run_dir()

    # write the current mrae head commit
    utils.write_repo_commit_hash(os.path.join(run_dir,'commit_state.txt'))

    # load hyperparameter, save to run directory
    hparam_read_file = os.path.join(CURR_DIR,'hyperparameters','run_example.yaml')
    hparams = utils.read_yaml(hparam_read_file)
    hparam_write_file = os.path.join(run_dir,'hyperparameters.yaml')
    utils.write_yaml(hparams,hparam_write_file)

    # create dataset and dataloaders (3-band dataset for 3-block model)
    ecog_ds = utils.get_test_ecog_dataset(device='cpu')
    train_dl, valid_dl, test_dl = data.get_partition_dataloaders(ecog_ds,batch_size=BATCH_SIZE)

    # initialize model, objective from hyperparameter yaml file, dataset sizes
    mrae = mrae_model_from_hparams(ecog_ds.input_size,hparams)
    mrae_obj = mrae_obj_from_hparams(hparams)

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

if __name__ == "__main__":
    main(None)