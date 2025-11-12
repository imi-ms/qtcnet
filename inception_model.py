import torch.nn as nn
from tsai.models.InceptionTimePlus import InceptionTimePlus


class ECGModel(nn.Module):
    def __init__(self, c_in=12, c_out=1, **kwargs):
        super(ECGModel, self).__init__()
        self.model = InceptionTimePlus(c_in=c_in, c_out=c_out, **kwargs)

    def forward(self, x):
        # The data should be in [batch_size, num_channels, time_steps] shape
        return self.model(x)
