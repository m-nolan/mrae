import unittest
import torch
from mrae import rnn

class RNNTests(unittest.TestCase):

    def test_gru_cell(self):
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