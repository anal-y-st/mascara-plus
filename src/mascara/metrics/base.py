
import torch
class Metric:
    def reset(self): raise NotImplementedError
    def update(self, preds: torch.Tensor, target: torch.Tensor): raise NotImplementedError
    def compute(self) -> float: raise NotImplementedError
class IoU(Metric):
    def __init__(self, threshold=0.5, eps=1e-6): self.t=threshold; self.eps=eps; self.reset()
    def reset(self): self.inter=0.0; self.union=0.0
    @torch.no_grad()
    def update(self,preds,target): p=(torch.sigmoid(preds)>=self.t).float(); t=target.float(); self.inter+=(p*t).sum().item(); self.union+=(p+t-p*t).sum().item()
    def compute(self): return (self.inter+self.eps)/(self.union+self.eps)
class Dice(Metric):
    def __init__(self, threshold=0.5, eps=1e-6): self.t=threshold; self.eps=eps; self.reset()
    def reset(self): self.tp=0.0; self.fp=0.0; self.fn=0.0
    @torch.no_grad()
    def update(self,preds,target): p=(torch.sigmoid(preds)>=self.t).float(); t=target.float(); self.tp+=(p*t).sum().item(); self.fp+=(p*(1-t)).sum().item(); self.fn+=((1-p)*t).sum().item()
    def compute(self): return (2*self.tp + self.eps)/(2*self.tp + self.fp + self.fn + self.eps)
class PixelAccuracy(Metric):
    def __init__(self, threshold=0.5, eps=1e-6): self.t=threshold; self.eps=eps; self.reset()
    def reset(self): self.correct=0.0; self.count=0.0
    @torch.no_grad()
    def update(self,preds,target): p=(torch.sigmoid(preds)>=self.t).float(); t=target.float(); self.correct += (p==t).float().sum().item(); self.count += t.numel()
    def compute(self): return (self.correct + self.eps) / (self.count + self.eps)
