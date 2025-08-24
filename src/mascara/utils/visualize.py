
from pathlib import Path
import torch
from PIL import Image
def _to_img(t: torch.Tensor) -> Image.Image:
    if t.ndim == 3 and t.shape[0] in (1,3):
        if t.shape[0] == 1: t = t[0]
        else: t = t.permute(1,2,0)
    if t.ndim == 2:
        arr = (t.clamp(0,1)*255).to(torch.uint8).cpu().numpy()
        return Image.fromarray(arr, mode="L")
    arr = (t.clamp(0,1)*255).to(torch.uint8).cpu().numpy()
    return Image.fromarray(arr)
@torch.no_grad()
def save_triptych(logits: torch.Tensor, targets: torch.Tensor, out_dir: Path, prefix: str, step: int):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    probs = torch.sigmoid(logits).detach()
    for i in range(probs.shape[0]):
        pred = probs[i,0] if probs.shape[1] == 1 else probs[i].mean(0)
        gt   = targets[i,0] if targets.shape[1] == 1 else targets[i].mean(0)
        diff = (pred - gt).abs()
        pred_img = _to_img(pred); gt_img = _to_img(gt); diff_img = _to_img(diff)
        w = gt_img.width + pred_img.width + diff_img.width; h = max(gt_img.height,pred_img.height,diff_img.height)
        strip = Image.new("L",(w,h)); x=0
        for im in (gt_img,pred_img,diff_img):
            if im.height != h: im = im.resize((im.width,h))
            strip.paste(im,(x,0)); x+=im.width
        strip.save(out_dir / f"{prefix}_idx{i:03d}_step{step}.png")
