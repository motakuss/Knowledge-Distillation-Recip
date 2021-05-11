import torch
import torch.nn as nn

class SoftTargetLoss(nn.Module):
    '''
    Distilling the knowledge in a Neural Network
    '''
    def __init__(self, T=10):
        super().__init__()
        self.T = T
    
    def forward(self, logits, targets):
        '''
        logits: 予測結果(ネットワークの出力)
        targets: 正解
        '''
        logits = logits / self.T
        targets = targets / self.T
        loss = nn.KLDivLoss(reduction='batchmean')
        p = nn.Softmax(dim=1)
        q = nn.LogSoftmax(dim=1)
        kd_loss = loss(q(logits), p(targets))
        return kd_loss
        
if __name__ == '__main__':
    x = torch.randn((32,10))
    t = torch.randn((32,10))
    loss = SoftTargetLoss()
    kd_loss = loss(x, t)
    print(kd_loss)
    print(loss.T)
