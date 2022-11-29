import torch.nn as nn
import torch.nn.functional as F
import torch

class NIEnsembleNet(nn.Module):
    def __init__(self):
        super(NIEnsembleNet, self).__init__()
        self.classifier = nn.Linear(1920+68, 5)

    def forward(self, x1, x2):
        # print(x1.shape, x2.shape)
        x2 = x2.view(x2.shape[0], -1)
        x = torch.cat([x1, x2], dim=1)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    import numpy as np
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs_test = np.random.rand(2, 68, 1, 1, 1)
    inputs_test = torch.from_numpy(inputs_test)
    inputs_test = inputs_test.float()
    inputs_test = inputs_test.to(device)

    inputs_test1 = np.random.rand(2, 1920, 1, 1, 1)
    inputs_test1 = torch.from_numpy(inputs_test1)
    inputs_test1 = inputs_test1.float()
    inputs_test1 = inputs_test1.to(device)

    net_test = NIEnsembleNet().to(device)
    res = net_test(inputs_test, inputs_test1)

    print(res.shape)