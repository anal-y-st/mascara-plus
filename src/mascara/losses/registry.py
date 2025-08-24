
import torch.nn as nn
LOSS_REGISTRY = {"crossentropy": nn.BCEWithLogitsLoss, "bce": nn.BCEWithLogitsLoss, "mse": nn.MSELoss, "huber": nn.HuberLoss}
def get_loss(name: str, **kwargs):
    key = name.lower()
    if key not in LOSS_REGISTRY: raise ValueError(f"Unknown loss '{name}'")
    return LOSS_REGISTRY[key](**kwargs)
class CombinedLoss(nn.Module):
    def __init__(self, losses):
        super().__init__()
        if not losses: raise ValueError("CombinedLoss requires losses list")
        self.fns = nn.ModuleList([get_loss(n) for n,_ in losses])
        self.ws = [float(w) for _,w in losses]
    def forward(self,preds,targets):
        tot = 0.0
        for w,fn in zip(self.ws,self.fns): tot = tot + w * fn(preds, targets.float())
        return tot
