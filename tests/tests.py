from msilib import sequence
import unittest
import torch
from mrae import MRAE, objective, rnn

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

    def test_objective_and_optimizers(self):
        # do a forward pass of the model then a forward pass of the objective class, check outputs
        input_size = 10
        num_blocks = 5
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

        kl_div_scale = 0.2
        l2_scale = 0.2
        mrae_obj = objective.MRAEObjective(
            kl_div_scale = kl_div_scale,
            l2_scale = l2_scale
        )

        # forward pass
        batch_size = 40
        sequence_length = 50
        input = torch.randn(batch_size,sequence_length,input_size,num_blocks)
        target = torch.randn(batch_size,sequence_length,input_size)
        mrae_output = mrae(input)

        output_obj, block_obj = mrae_obj(mrae_output,target,input)

        self.assertTrue(output_obj > 0)
        self.assertEqual(output_obj.numel(),1)
        self.assertTrue((block_obj > 0).all())
        self.assertEqual(block_obj.numel(),num_blocks)

    def test_optimizer(self):
        # test parameter updates in optimizer backward pass for multiblock model
        # do a forward pass of the model then a forward pass of the objective class, check outputs
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

        kl_div_scale = 0.2
        l2_scale = 0.2
        mrae_obj = objective.MRAEObjective(
            kl_div_scale = kl_div_scale,
            l2_scale = l2_scale
        )

        # forward pass
        batch_size = 40
        sequence_length = 50
        input = torch.randn(batch_size,sequence_length,input_size,num_blocks)
        target = torch.randn(batch_size,sequence_length,input_size)
        mrae_output = mrae(input)

        output_obj, block_obj = mrae_obj(mrae_output,target,input)
        mrae.configure_optimizers()

        # backward pass over blocks, then output - check 1-block case with output ID layer
        # model backward pass call includes zero_grad and clip calls
        mrae.backward(output_obj, block_obj)

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

        kl_div_scale = 0.2
        l2_scale = 0.2
        mrae_obj = objective.MRAEObjective(
            kl_div_scale = kl_div_scale,
            l2_scale = l2_scale
        )

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