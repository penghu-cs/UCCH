from torch import nn

class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        # self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.softmax(1)
        return -x[:, 0].log().mean()