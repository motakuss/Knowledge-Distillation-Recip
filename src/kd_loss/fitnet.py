import torch
import torch.nn as nn

class HintLearningLoss(nn.Module):
    '''
    FitNets: Hint for Thin Deep Nets
    '''
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        '''
        logits: 生徒モデルの中間層出力
        targets: 教師モデルの中間層出力
        '''
        loss = nn.MSELoss()
        kd_loss = loss(logits, targets)
        return kd_loss

if __name__ == '__main__':
    x = torch.randn((10,32,32,3))
    t = torch.randn((10,32,32,3))
    loss = HintLearningLoss()
    kd_loss = loss(x,t)
    print(kd_loss)
