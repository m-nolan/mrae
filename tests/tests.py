import unittest
import os
from nbformat import write
import torch
import h5py
from mrae import data, MRAE, objective, rnn

test_dir = os.path.dirname(__file__)
data_dir = os.path.join(test_dir,'data')
write_dir = os.path.join(test_dir,'tmp')
if not os.path.exists(write_dir):
    os.mkdir(write_dir)

class RAETests(unittest.TestCase):
    
    def test_mrae(self):
        input_size = 10
        num_blocks = 1
        encoder_size = 20
        decoder_size = 20
        dropout = 0.3

        mrae = MRAE.MRAE(
            input_size=input_size,
            encoder_size=encoder_size,
            decoder_size=decoder_size,
            num_blocks=num_blocks,
            dropout=dropout
        )

        # forward pass
        batch_size = 40
        sequence_length = 50
        input = torch.randn(batch_size,sequence_length,input_size,num_blocks)
        mrae_output = mrae(input)

        self.assertEqual(mrae_output.output.size(),(batch_size,sequence_length,input_size))
        self.assertEqual(mrae_output.block_output.size(),(batch_size,sequence_length,input_size,num_blocks))
        self.assertEqual(mrae_output.hidden.size(),(batch_size,sequence_length,decoder_size,num_blocks))
        self.assertEqual(mrae_output.decoder_ic.size(),(batch_size,decoder_size,num_blocks))
        self.assertEqual(mrae_output.decoder_ic_kl_div.numel(),num_blocks)
        self.assertEqual(mrae_output.decoder_l2.numel(),num_blocks)

    def test_rae_block(self):
        input_size = 10
        encoder_size = 20
        decoder_size = 30
        dropout = 0.3

        rae_block = MRAE.RAE_block(
            input_size=input_size,
            encoder_size=encoder_size,
            decoder_size=decoder_size,
            dropout=dropout
        )

        batch_size = 50
        sequence_length = 40
        input = torch.randn(batch_size,sequence_length,input_size)

        block_out, block_hidden, block_dec_ic, block_kl_div = rae_block(input)

        self.assertEqual(block_out.size(),(batch_size,sequence_length,input_size))
        self.assertEqual(block_hidden.size(),(batch_size,sequence_length,decoder_size))
        self.assertEqual(block_dec_ic.size(),(batch_size,decoder_size))
        self.assertTrue(block_kl_div > 0)

    def test_encoder(self):
        input_size = 10
        output_size = 10
        hidden_size = 30
        enc_dropout = 0.3
        num_layers = 2 # try more layers later
        bidirectional = True # try false, you won't

        enc = MRAE.Encoder(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            dropout=enc_dropout,
            num_layers=num_layers,
            bidirectional=bidirectional
        )

        batch_size = 5
        sequence_length = 20
        input = torch.randn(batch_size,sequence_length,input_size)

        mean, logvar = enc(input)
        self.assertEqual(mean.size(),(batch_size,output_size))
        self.assertEqual(logvar.size(),(batch_size,output_size))

    def test_decoder(self):
        hidden_size = 10
        dec_dropout = 0.3
        num_layers = 1
        bidirectional = False

        dec = MRAE.Decoder(
            hidden_size=hidden_size,
            dropout=dec_dropout,
            num_layers=num_layers,
            bidirectional=bidirectional
        )

        batch_size = 8
        sequence_length = 50
        input = dec.gen_input(batch_size,sequence_length)
        h0 = torch.randn(batch_size,hidden_size)

        dec_out = dec(input,h0)

        self.assertEqual(dec_out.size(),(batch_size,sequence_length,hidden_size))

class OptimizationTests(unittest.TestCase):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        # do a forward pass of the model then a forward pass of the objective class, check outputs
        self.input_size = 10
        self.num_blocks = 3
        batch_size = 10
        self.encoder_size = 20
        self.decoder_size = 20
        self.dropout = 0.3

        self.mrae = MRAE.MRAE(
            input_size=self.input_size,
            encoder_size=self.encoder_size,
            decoder_size=self.decoder_size,
            num_blocks=self.num_blocks,
            dropout=self.dropout
        )

        kl_div_scale_max = 0.2
        l2_scale_max = 0.2
        self.mrae_obj = objective.MRAEObjective(
            kl_div_scale_max = kl_div_scale_max,
            l2_scale_max = l2_scale_max
        )
        self.mrae.configure_optimizers()
        self.mrae.configure_schedulers()

        # testing dataset + training loop
        target_dataset_path = os.path.join(data_dir,'gw_250_test')
        band_dataset_path = os.path.join(data_dir,f'gw_250_nband{self.num_blocks}_test')
        target_data_record = h5py.File(target_dataset_path,'r')
        band_data_record = h5py.File(band_dataset_path,'r')
        ds = data.MultiblockTensorDataset(
            target_data_record=target_data_record,
            band_data_record=band_data_record,
            n_band = self.num_blocks
        )
        self.train_dl, self.valid_dl, self.test_dl = data.get_partition_dataloaders(
            ds=ds,
            batch_size=batch_size
        )

        self.mrae_ecog = MRAE.MRAE(
            input_size=ds.target_data_record['ecog'].shape[-1],
            encoder_size=self.encoder_size,
            decoder_size=self.decoder_size,
            num_blocks=self.num_blocks,
            dropout=self.dropout,
        )

    def test_checkpoint(self):
        model_checkpoint_file_path = os.path.join(write_dir,'test_model.pt')
        obj_checkpoint_file_path = os.path.join(write_dir,'test_objective.pt')
        epoch = 0
        test_training_loss = 5.0
        test_validation_loss = 5.0
        search_count = 0
        
        # test save
        self.mrae.save_checkpoint(
            model_checkpoint_file_path,
            epoch,
            test_training_loss,
            test_validation_loss,
            search_count,
        )
        torch.save(self.mrae_obj.state_dict(),obj_checkpoint_file_path)

        # test loading
        mrae_test = MRAE.MRAE(self.input_size,self.encoder_size,self.decoder_size,self.num_blocks,self.dropout)
        # note: the whole hyperparameterization needs to line up, so I need to save those out in a .yaml file as well
        mrae_test.load_mrae_checkpoint(model_checkpoint_file_path)
        mrae_obj_test_state_dict = torch.load(obj_checkpoint_file_path)
        mrae_obj_test = objective.MRAEObjective(1.0,1.0)
        mrae_obj_test.load_state_dict(mrae_obj_test_state_dict)

        self.assertAlmostEqual(mrae_obj_test.kl_div_scale,self.mrae_obj.kl_div_scale)
        self.assertTrue(
            (mrae_test.state_dict()['block_hidden_mix.weight'] == 
            self.mrae.state_dict()['block_hidden_mix.weight']).all()
        )
        # self.assertDictEqual(mrae_obj_test.state_dict(),self.mrae_obj.state_dict())

    def test_objective_and_optimizers(self):
        # do a forward pass of the model then a forward pass of the objective class, check outputs
        batch_size = 40
        sequence_length = 50
        input = torch.randn(batch_size,sequence_length,self.input_size,self.num_blocks)
        target = torch.randn(batch_size,sequence_length,self.input_size)
        mrae_output = self.mrae(input)

        output_obj, block_obj = self.mrae_obj(mrae_output,target,input)

        self.assertTrue(output_obj > 0)
        self.assertEqual(output_obj.numel(),1)
        self.assertTrue((block_obj > 0).all())
        self.assertEqual(block_obj.numel(),self.num_blocks)

    def test_optimizer(self):
        # test parameter updates in optimizer backward pass for multiblock model
        # do a forward pass of the model then a forward pass of the objective class, check outputs
        # forward pass
        batch_size = 40
        sequence_length = 50
        input = torch.randn(batch_size,sequence_length,self.input_size,self.num_blocks)
        target = torch.randn(batch_size,sequence_length,self.input_size)
        mrae_output = self.mrae(input)

        output_obj, block_obj = self.mrae_obj(mrae_output,target,input)

        # backward pass over blocks, then output - check 1-block case with output ID layer
        # model backward pass call includes zero_grad and clip calls
        self.mrae.backward(output_obj, block_obj)
        #TODO: add an assert call here, not it's just "passes if it doesn't crash" and that's not enough

    def test_train_step(self):
        batch_size = 40
        sequence_length = 50
        epoch_idx = 0
        input = torch.randn(batch_size,sequence_length,self.input_size,self.num_blocks)
        target = torch.randn(batch_size,sequence_length,self.input_size)

        self.mrae.train()
        mrae_output, train_output_loss, train_block_loss = self.mrae.train_step(
            epoch_idx, input, target, self.mrae_obj
        )

        # check loss outputs
        self.assertTrue(train_output_loss > 0)
        self.assertEqual(train_output_loss.numel(), 1)
        self.assertTrue((train_block_loss > 0).all())
        self.assertEqual(train_block_loss.numel(),self.num_blocks)

    def test_valid_step(self):
        batch_size = 40
        sequence_length = 50
        epoch_idx = 10 # causes checkable obj param update in .valid_step()
        input = torch.randn(batch_size,sequence_length,self.input_size,self.num_blocks)
        target = torch.randn(batch_size,sequence_length,self.input_size)

        self.mrae.eval()
        mrae_output, valid_output_loss, valid_block_loss = self.mrae.valid_step(
            epoch_idx, input, target, self.mrae_obj
        )

        # check loss outputs
        self.assertTrue(valid_output_loss > 0)
        self.assertEqual(valid_output_loss.numel(), 1)
        self.assertTrue((valid_block_loss > 0).all())
        self.assertEqual(valid_block_loss.numel(),self.num_blocks)

        # check scheduler weight updates
        epoch = epoch_idx+1
        self.assertAlmostEqual(
            self.mrae_obj.kl_div_scale,
            self.mrae_obj.kl_div_scale_max * epoch / self.mrae_obj.max_at_epoch
        )
        self.assertAlmostEqual(
            self.mrae_obj.l2_scale,
            self.mrae_obj.l2_scale_max * epoch / self.mrae_obj.max_at_epoch
        )

    def test_scheduler(self):
        # create scheduler to update objective parameters, test update
        input_size = 10
        num_blocks = 1
        encoder_size = 20
        decoder_size = 20
        dropout = 0.3
        max_grad_norm = 3.0

        mrae = MRAE.MRAE(
            input_size=input_size,
            encoder_size=encoder_size,
            decoder_size=decoder_size,
            num_blocks=num_blocks,
            dropout=dropout,
            max_grad_norm=max_grad_norm
        )

        kl_div_scale_max = 0.2
        l2_scale_max = 0.2
        max_at_epoch = 10
        mrae_obj = objective.MRAEObjective(
            kl_div_scale_max = kl_div_scale_max,
            l2_scale_max = l2_scale_max,
            max_at_epoch = max_at_epoch
        )

        epoch_idx = 0

        # forward pass
        batch_size = 40
        sequence_length = 50
        input = torch.randn(batch_size,sequence_length,input_size,num_blocks)
        target = torch.randn(batch_size,sequence_length,input_size)
        mrae_output = mrae(input)

        output_obj, block_obj = mrae_obj(mrae_output,target,input)
        mrae.configure_optimizers()
        mrae.configure_schedulers()

        # backward pass over blocks, then output - check 1-block case with output ID layer
        # model backward pass call includes zero_grad and clip calls
        mrae.backward(output_obj, block_obj)
        mrae.step_schedulers(output_obj, block_obj)

        # test objective parameter update
        epoch_idx += 1
        mrae_obj.step(epoch_idx)
        epoch = epoch_idx + 1

        self.assertAlmostEqual(mrae_obj.kl_div_scale,kl_div_scale_max * epoch / max_at_epoch)
        self.assertAlmostEqual(mrae_obj.l2_scale,l2_scale_max * epoch / max_at_epoch)

    def test_fit(self):
        max_epochs = 2
        min_epochs = 0
        test_run_dir = os.path.join(write_dir,'test_fit')
        self.mrae_ecog.configure_optimizers()
        self.mrae_ecog.configure_schedulers()
        self.mrae_ecog.fit(
            self.train_dl,
            self.valid_dl,
            self.mrae_obj,
            save_dir=test_run_dir,
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            overwrite=True
        )

    def test_evaluate(self):
        pass

class DataloaderTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_band = 3
        target_dataset_path = os.path.join(data_dir,'gw_250_test')
        band_dataset_path = os.path.join(data_dir,f'gw_250_nband{self.n_band}_test')
        target_data_record = h5py.File(target_dataset_path,'r')
        band_data_record = h5py.File(band_dataset_path,'r')
        self.num_trials, self.sequence_length, self.num_ch = target_data_record['ecog'].shape

        # base dataset for multiband ecog datasets
        self.ds = data.MultiblockTensorDataset(
            target_data_record=target_data_record,
            band_data_record=band_data_record,
            n_band=self.n_band,
        )

        # train, valid, and test dataloaders
        self.batch_size = 10
        self.train_dl, self.valid_dl, self.test_dl = data.get_partition_dataloaders(
            ds=self.ds,
            batch_size=self.batch_size
        )

    def test_multiblock_tensor_dataset(self):
        input, target = self.ds[0]
        
        self.assertEqual(input.size(),(self.sequence_length,self.num_ch,self.n_band))
        self.assertEqual(target.size(),(self.sequence_length,self.num_ch))

    def test_partition_dataloader(self):
        #TODO: fix this weird added leading dimension bug
        input_train_batch, target_train_batch = next(iter(self.train_dl))
        input_train_batch = input_train_batch.squeeze()
        target_train_batch = target_train_batch.squeeze()
        input_valid_batch, target_valid_batch = next(iter(self.valid_dl))
        input_valid_batch = input_valid_batch.squeeze()
        target_valid_batch = target_valid_batch.squeeze()
        input_test_batch, target_test_batch = next(iter(self.test_dl))
        input_test_batch = input_test_batch.squeeze()
        target_test_batch = target_test_batch.squeeze()

        self.assertEqual(
            input_train_batch.size(),
            (self.batch_size,self.sequence_length,self.num_ch,self.n_band)
        )
        self.assertEqual(
            input_valid_batch.size(),
            (self.batch_size,self.sequence_length,self.num_ch,self.n_band)
        )
        self.assertEqual(
            input_test_batch.size(),
            (self.batch_size,self.sequence_length,self.num_ch,self.n_band)
        )
        self.assertEqual(
            target_train_batch.size(),
            (self.batch_size,self.sequence_length,self.num_ch)
        )
        self.assertEqual(
            target_valid_batch.size(),
            (self.batch_size,self.sequence_length,self.num_ch)
        )
        self.assertEqual(
            target_test_batch.size(),
            (self.batch_size,self.sequence_length,self.num_ch)
        )

class RNNTests(unittest.TestCase):

    def test_gru_modified(self):
        batch_size = 1
        sequence_length = 20
        input_size = 10
        hidden_size = 10

        test_gru = rnn.GRU_Modified(
            input_size=input_size,
            hidden_size=hidden_size
        )

        test_input = torch.randn(batch_size,sequence_length,input_size)

        test_output = test_gru(
            test_input
        )

        self.assertEqual(test_output.size(),(batch_size,sequence_length,input_size))

    def test_gru_cell_modified(self):
        batch_size = 1
        input_size = 10
        hidden_size = 10
        update_bias = 1.0

        test_gru_cell = rnn.GRU_Cell_Modified(
            input_size=input_size,
            hidden_size=hidden_size,
            update_bias=update_bias
        )

        test_input = torch.randn(batch_size,input_size)
        test_hidden = torch.randn(batch_size,hidden_size)
        
        test_hidden_update = test_gru_cell(
            test_input,
            test_hidden
        )

        self.assertEqual(test_hidden_update.size(),(batch_size,hidden_size))

if __name__ == "__main__":
    unittest.main()
