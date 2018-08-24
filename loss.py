import torch
import torch.nn as nn

class JointLoss(nn.Module): 
    def __init__(self, n_out=3):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_out))
        self.params = nn.ParameterList([self.weights])
        self.sm = nn.Softmax()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, y):
        loss = 0
        w = self.sm(self.weights)
        for w_, p_, y_ in zip(w, pred, y):
            loss += w_ * self.ce(p_, y_)
        return loss

class StaticLoss(nn.Module): 
    def __init__(self, n_out=3, stl_dim=-1):
        super().__init__()

        if stl_dim == -1: 
            self.weights = nn.Parameter(torch.ones(n_out)/n_out, requires_grad=False)
        else:
            weights = torch.zeros(n_out)
            weights[stl_dim] = 1
            self.weights = nn.Parameter(weights, requires_grad=False)
        self.params = nn.ParameterList([self.weights])
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, y):
        loss = 0
        for w_, p_, y_ in zip(self.weights, pred, y):
            loss += w_ * self.ce(p_, y_)
        return loss

def get_loss(loss, n_out=3, stl_dim=-1):
    if loss=='Joint':
        return JointLoss(n_out)
    elif loss=='Static':
        return StaticLoss(n_out, stl_dim)
    else:
        raise ValueError('Invalid Loss Type')
