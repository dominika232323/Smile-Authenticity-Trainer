import torch
import pytest

from modeling.smile_net import SmileNet


class TestSmileNet:
    def test_smile_net_initialization_default(self):
        """Test if SmileNet initializes with default hidden layers."""
        input_dim = 10
        model = SmileNet(input_dim=input_dim)

        # Check if the internal sequential model is created
        assert isinstance(model.net, torch.nn.Sequential)
        # Default is [128, 64] -> (Linear, ReLU, Dropout) * 2 + Final Linear = 7 layers
        assert len(model.net) == 7

    def test_smile_net_forward_shape(self):
        """Test if the forward pass returns the correct output shape."""
        input_dim = 20
        batch_size = 8
        model = SmileNet(input_dim=input_dim, hidden_dims=[32])

        x = torch.randn(batch_size, input_dim)
        output = model(x)

        # squeeze(1) results in (batch_size,) if the final linear output was (batch_size, 1)
        assert output.shape == (batch_size,)

    def test_smile_net_custom_hidden_dims(self):
        """Test if SmileNet respects custom hidden dimensions."""
        input_dim = 10
        hidden_dims = [16, 8, 4]
        model = SmileNet(input_dim=input_dim, hidden_dims=hidden_dims)

        # (Linear, ReLU, Dropout) * 3 + 1 Final Linear = 10 layers
        assert len(model.net) == 10

        # Verify specific layer dimensions
        assert model.net[0].out_features == 16
        assert model.net[3].out_features == 8
        assert model.net[-1].out_features == 1

    def test_smile_net_dropout_behavior(self):
        """Test if dropout is active during training and inactive during evaluation."""
        model = SmileNet(input_dim=10, dropout_p=0.5)
        x = torch.randn(5, 10)

        model.train()
        out1 = model(x)
        out2 = model(x)

        # With dropout 0.5, outputs are highly likely to be different in train mode
        assert not torch.equal(out1, out2)

        model.eval()
        out_eval1 = model(x)
        out_eval2 = model(x)

        # In eval mode, dropout is disabled, outputs must be identical
        assert torch.equal(out_eval1, out_eval2)

    @pytest.mark.parametrize("batch_size", [1, 16, 32])
    def test_smile_net_various_batches(self, batch_size):
        """Test forward pass with different batch sizes."""
        input_dim = 5
        model = SmileNet(input_dim=input_dim)
        x = torch.randn(batch_size, input_dim)
        output = model(x)
        assert output.shape[0] == batch_size
