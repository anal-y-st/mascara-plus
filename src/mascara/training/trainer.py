
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from ..metrics import Metric
from ..utils.visualize import save_triptych
from ..logging_utils import setup_logging

def _device(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")
def _check_batch(x: torch.Tensor, y: torch.Tensor):
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor): raise TypeError("Each batch must be a pair of torch.Tensor (x, y).")
    if x.ndim != 4 or y.ndim != 4: raise ValueError(f"Expected x,y (B,C,H,W) and (B,1,H,W). Got {x.shape}, {y.shape}.")
    if y.shape[1] != 1: raise ValueError(f"Target must have 1 channel, got {y.shape}.")

class Trainer:
    def __init__(self, model: torch.nn.Module, optimizer_cfg: Dict[str,Any]|None=None, scheduler_cfg: Dict[str,Any]|None=None, metrics: List[Metric]|None=None, workdir: str = "outputs/run", trainer_cfg: Dict[str,Any]|None=None, loss_fn: Optional[torch.nn.Module]=None, device: Optional[str]=None):
        self.model = model; self.device = torch.device(device) if device else _device(); self.model.to(self.device)
        opt_name = (optimizer_cfg or {}).get("name","adam").lower(); lr = float((optimizer_cfg or {}).get("lr",1e-3)); weight_decay = float((optimizer_cfg or {}).get("weight_decay",0.0)); momentum = float((optimizer_cfg or {}).get("momentum",0.9))
        sch_name = (scheduler_cfg or {}).get("name", None); sch_name = sch_name.lower() if sch_name else None
        T_max = int((scheduler_cfg or {}).get("T_max",50)); step_size = int((scheduler_cfg or {}).get("step_size",30)); gamma = float((scheduler_cfg or {}).get("gamma",0.1))
        if opt_name == "adam": self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == "adamw": self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == "sgd": self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else: raise ValueError(f"Unknown optimizer: {opt_name}")
        if sch_name == "cosine": self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif sch_name == "step": self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif sch_name == "plateau": self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, factor=gamma)
        else: self.scheduler = None
        self.loss_fn = loss_fn or torch.nn.BCEWithLogitsLoss(); self.metrics = metrics or []
        self.workdir = Path(workdir); self.ckpt_dir = self.workdir/"checkpoints"; self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir = self.workdir/"viz"; self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logging(str(self.workdir)); self.writer = SummaryWriter(log_dir=str(self.workdir/"tb"))
        self.amp = bool((trainer_cfg or {}).get("amp", True)); self.grad_clip = (trainer_cfg or {}).get("grad_clip", None)
        self.viz_every = int((trainer_cfg or {}).get("viz_every_n_steps", 200)); self.log_every = int((trainer_cfg or {}).get("log_every_n_steps", 50)); self.max_viz = int((trainer_cfg or {}).get("max_viz_items",4)); self.early_patience = (trainer_cfg or {}).get("early_stop_patience", 10)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp and self.device.type=="cuda")
        self.best_score = float("-inf"); self.early_counter = 0; self.logger.info(f"Device: {self.device} | AMP: {self.amp and self.device.type=='cuda'}")
    def _run_metrics(self, preds, targets, stage: str, step: int):
        for m in self.metrics: m.update(preds, targets); self.writer.add_scalar(f"{stage}/{m.__class__.__name__}", m.compute(), step)
    def _reset_metrics(self): 
        for m in self.metrics: m.reset()
    def _len_safe(self, loader):
        try: return len(loader)
        except Exception: return None
    def fit(self, train_loader, val_loader=None, epochs: int = 1, log_images: bool = True):
        step = 0
        for epoch in range(1, epochs+1):
            self.model.train(); self._reset_metrics()
            pbar = tqdm(train_loader, total=self._len_safe(train_loader), desc=f"Train {epoch}", dynamic_ncols=True)
            for i, batch in enumerate(pbar, start=1):
                if isinstance(batch,(list,tuple)) and len(batch)==2: x,y = batch
                else: raise TypeError("Each batch must be (x,y)")
                _check_batch(x,y); x = x.to(self.device, non_blocking=(self.device.type=='cuda')); y = y.to(self.device, non_blocking=(self.device.type=='cuda'))
                self.optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.amp and self.device.type=='cuda'):
                    logits = self.model(x); loss = self.loss_fn(logits, y)
                self.scaler.scale(loss).backward()
                if self.grad_clip: self.scaler.unscale_(self.optimizer); torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer); self.scaler.update()
                if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau): self.scheduler.step()
                self.writer.add_scalar("train/loss", loss.item(), step); pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                if (i % self.log_every) == 0: self.logger.info(f"epoch {epoch} step {i} loss={loss.item():.4f}")
                self._run_metrics(logits.detach(), y, stage="train", step=step)
                if log_images and (i % self.viz_every) == 0: b = min(self.max_viz, x.shape[0]); save_triptych(logits[:b].detach(), y[:b], self.viz_dir, prefix=f"train_ep{epoch}_it{i}", step=step)
                step += 1
            if val_loader is not None:
                score = self.validate(val_loader, epoch=epoch) or -loss.item()
                if score > self.best_score: self.best_score = score; self.early_counter = 0; self._save_checkpoint(epoch, True)
                else: self.early_counter += 1; 
                if self.early_patience and self.early_counter >= self.early_patience: self.logger.info("Early stopping."); break
        self._save_checkpoint(epoch, False)
    @torch.no_grad()
    def validate(self, loader, epoch: int = 0):
        self.model.eval(); self._reset_metrics(); losses=[]
        pbar = tqdm(loader, total=self._len_safe(loader), desc=f"Val   {epoch}", dynamic_ncols=True)
        for i, batch in enumerate(pbar, start=1):
            if isinstance(batch,(list,tuple)) and len(batch)==2: x,y = batch
            else: raise TypeError("Each batch must be (x,y)")
            _check_batch(x,y); x = x.to(self.device); y = y.to(self.device)
            logits = self.model(x); loss = self.loss_fn(logits, y); losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"}); self._run_metrics(logits, y, stage="val", step=i+epoch)
            if (i % self.viz_every) == 0: b = min(self.max_viz, x.shape[0]); save_triptych(logits[:b], y[:b], self.viz_dir, prefix=f"val_ep{epoch}_it{i}", step=i+epoch)
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau): self.scheduler.step(sum(losses)/len(losses))
        return self.metrics[0].compute() if self.metrics else None
    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval(); self._reset_metrics()
        pbar = tqdm(loader, total=self._len_safe(loader), desc="Test  ", dynamic_ncols=True)
        for i, batch in enumerate(pbar, start=1):
            if isinstance(batch,(list,tuple)) and len(batch)==2: x,y = batch
            else: raise TypeError("Each batch must be (x,y)")
            _check_batch(x,y); x = x.to(self.device); y = y.to(self.device); logits = self.model(x); self._run_metrics(logits, y, stage="test", step=i)
        return {m.__class__.__name__: m.compute() for m in self.metrics}
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        path = Path(self.workdir)/"checkpoints"/("best.pth" if is_best else f"last_ep{epoch}.pth"); path.parent.mkdir(parents=True, exist_ok=True)
        import torch; torch.save(self.model.state_dict(), path); self.logger.info(f"Saved weights to {path}")
