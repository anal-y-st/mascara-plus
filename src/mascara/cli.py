
from typing import Tuple, Optional, Dict, Any
import importlib, json, yaml
from pathlib import Path
import typer
from .models import MODEL_REGISTRY
from .training import Trainer
from .metrics import Dice, IoU, PixelAccuracy
from .losses import get_loss, CombinedLoss

app = typer.Typer(help="mascara: mask training & evaluation")

def parse_kv(s: Optional[str]):
    if not s: return {}
    p = Path(s)
    if p.exists():
        if p.suffix.lower() in {".yml",".yaml"}: return yaml.safe_load(p.read_text())
        if p.suffix.lower() == ".json": return json.loads(p.read_text())
        return yaml.safe_load(p.read_text())
    try: return json.loads(s)
    except json.JSONDecodeError: return yaml.safe_load(s)

def load_dataloaders(module_path: str, fn_name: str, **kwargs):
    mod = importlib.import_module(module_path); fn = getattr(mod, fn_name); ret = fn(**kwargs)
    from .utils.loader_adapter import adapt_user_getters
    return adapt_user_getters(ret)

@app.command()
def available_models():
    for k in MODEL_REGISTRY.keys(): typer.echo(k)

@app.command()
def train(dataloaders_module: str = typer.Option(...), get_loaders_fn: str = "get_dataloaders", model: str = "UNet", in_ch: int = 3, out_ch: int = 1, img_size: Tuple[int,int] = (128,128), embed_dim: int = 120, depth: int = 3, heads: int = 5, epochs: int = 10, batch_size: int = 8, lr: float = 1e-3, weight_decay: float = 1e-4, optimizer: str = "adam", scheduler: str = "cosine", workdir: str = "outputs/run1", viz_every: int = 200, log_every: int = 50, amp: bool = True, grad_clip: Optional[float] = None, loaders_kwargs: Optional[str] = None, loss: str = "crossentropy", loss_combo: Optional[str] = None):
    if model not in MODEL_REGISTRY: raise typer.BadParameter(f"Unknown model '{model}'. Available: {list(MODEL_REGISTRY)}")
    Model = MODEL_REGISTRY[model]
    if model == "AdvancedSwinUNet": model_inst = Model(in_ch=in_ch, out_ch=out_ch, embed_dim=embed_dim, depth=depth, heads=heads, img_size=tuple(img_size))
    else: model_inst = Model(in_ch=in_ch, out_ch=out_ch)
    if loss_combo: combos = json.loads(loss_combo); loss_fn = CombinedLoss(combos)
    else: loss_fn = get_loss(loss)
    dl_kwargs = parse_kv(loaders_kwargs) if loaders_kwargs else {}
    train_loader, val_loader, test_loader = load_dataloaders(dataloaders_module, get_loaders_fn, batch_size=batch_size, **dl_kwargs)
    trainer = Trainer(model=model_inst, optimizer_cfg={"name": optimizer, "lr": lr, "weight_decay": weight_decay}, scheduler_cfg={"name": None if scheduler=="none" else scheduler}, metrics=[Dice(), IoU(), PixelAccuracy()], workdir=workdir, trainer_cfg={"amp": amp, "viz_every_n_steps": viz_every, "log_every_n_steps": log_every, "grad_clip": grad_clip}, loss_fn=loss_fn)
    trainer.fit(train_loader, val_loader, epochs=epochs)
    if test_loader is not None: scores = trainer.evaluate(test_loader); typer.echo(f"Test: {scores}")
