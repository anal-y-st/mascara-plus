## mascara — Flexible segmentation training framework

Mascara is a lightweight framework for training image segmentation models (binary or multi-channel masks) with PyTorch. It provides ready-to-use models, metrics, logging, visualization, and an opinionated Trainer while remaining easy to extend.

### Features
- **UNet and Swin-based models** via a simple registry
- **BCE/CrossEntropy/MSE/Huber** losses and loss combinations
- **Metrics**: Dice, IoU, PixelAccuracy (TensorBoard logging included)
- **Visualizations**: GT | Pred | |pred-gt| triptychs saved as PNGs
- **Trainer** with AMP, schedulers, gradient clipping, early stopping
- **CLI and Python API** for quick experiments

## Installation

```bash
python -m venv venv
# Windows PowerShell
.\venv\Scripts\Activate.ps1
pip install -e .
```

Requires Python >= 3.9 and PyTorch.

## Quickstart (Python API)

The repository includes a minimal example dataset in `examples/loaders_template.py` and a runnable script in `main.py`.

Run training (and test evaluation):

```bash
python main.py
```

What this does:
- Builds `UNet(in_ch=3, out_ch=1)`
- Uses AdamW (lr=3e-4) and BCEWithLogitsLoss
- Logs metrics (Dice, IoU, PixelAccuracy) to TensorBoard under `outputs/unet_demo/tb`
- Saves model checkpoints under `outputs/unet_demo/checkpoints/`
- Saves triptych visualizations under `outputs/unet_demo/viz/`

Key knob in `main.py`:
- `viz_epochs`: only visualize the first N epochs (set to `None` to visualize every epoch)

## Quickstart (CLI)

Once installed with `pip install -e .`, you can use the `mascara` CLI:

```bash
mascara train \
  --dataloaders-module examples/loaders_template.py \
  --get-loaders-fn get_dataloaders \
  --model UNet \
  --in-ch 3 --out-ch 1 \
  --epochs 5 --batch-size 8 \
  --optimizer adamw --lr 0.0003 --scheduler none \
  --workdir outputs/unet_cli_demo \
  --viz-every 200 --log-every 5 \
  --loss bce
```

Notes:
- `--dataloaders-module` accepts either a module path (e.g., `my_pkg.my_loaders`) or a `.py` file path (e.g., `examples/loaders_template.py`).
- The CLI does not currently expose `viz_epochs`; use `main.py` to limit visualization by epoch count.

## Data loading

Provide a function that returns three dataloaders `(train, val, test)` or `(train, val, None)`:

```python
from torch.utils.data import DataLoader, Dataset

class RandomMaskDataset(Dataset):
    def __len__(self): ...
    def __getitem__(self, idx):
        # Return: (x: Tensor[C,H,W], y: Tensor[1,H,W])
        return x, y

def get_dataloaders(batch_size: int = 8, **kwargs):
    train = DataLoader(RandomMaskDataset(...), batch_size=batch_size, shuffle=True)
    val   = DataLoader(RandomMaskDataset(...), batch_size=batch_size)
    test  = DataLoader(RandomMaskDataset(...), batch_size=batch_size)
    return train, val, test
```

The framework adapts your return automatically; batches must be `(x, y)` where `x` and `y` are tensors with shapes `(B, C, H, W)` and `(B, 1, H, W)` respectively.

## Visualization

Triptychs are saved by `save_triptych` as three side-by-side panels:
- Ground Truth (GT)
- Predicted probability (after sigmoid)
- Absolute difference |pred - gt|

Control via `Trainer.trainer_cfg`:
- `viz_every_n_steps`: save images every N steps
- `max_viz_items`: number of items per save (default 4)
- `viz_epochs`: visualize only during the first N epochs (`None` = all epochs)

Outputs go to `<workdir>/viz/`.

## Trainer API (essentials)

```python
from mascara.training import Trainer

trainer = Trainer(
    model=model,
    optimizer_cfg={"name": "adamw", "lr": 3e-4, "weight_decay": 1e-4},
    scheduler_cfg={"name": "cosine"},  # or "step", "plateau", or None
    metrics=[Dice(), IoU(), PixelAccuracy()],
    workdir="outputs/run",
    trainer_cfg={
        "amp": True,
        "grad_clip": None,
        "viz_every_n_steps": 200,
        "log_every_n_steps": 50,
        "max_viz_items": 4,
        "early_stop_patience": 10,
        "viz_epochs": 2,  # None for all epochs
    },
    loss_fn=get_loss("bce"),
)

trainer.fit(train_loader, val_loader, epochs=5)
scores = trainer.evaluate(test_loader)  # dict of metrics
```

Batch requirements in the trainer:
- Inputs: `x` must be a float tensor `(B, C, H, W)`
- Targets: `y` must be a float tensor `(B, 1, H, W)`

## Models

Models are registered in `mascara.models.MODEL_REGISTRY`. Available keys include:
- `UNet`
- `AdvancedSwinUNet` (with extra params: `embed_dim`, `depth`, `heads`, `img_size`)

Example:

```python
from mascara.models import MODEL_REGISTRY
Model = MODEL_REGISTRY["UNet"]
model = Model(in_ch=3, out_ch=1)
```

## Losses

From `mascara.losses.get_loss(name)` where `name` in `{ "crossentropy", "bce", "mse", "huber" }`.

You can also combine losses in the CLI via `--loss-combo` (JSON list of `[name, weight]`).

## Metrics

Use `Dice`, `IoU`, `PixelAccuracy`. These are accumulated each epoch and logged to TensorBoard.

## Project structure

```
src/mascara/
  cli.py                # Typer CLI entrypoint
  training/trainer.py   # Trainer class (fit/validate/evaluate)
  models/               # UNet, AdvancedSwinUNet, registry
  losses/               # Loss registry and CombinedLoss
  metrics/              # Metric base and implementations
  utils/visualize.py    # save_triptych (GT | Pred | Diff)
examples/               # Example dataloaders
main.py                 # Python entry for quick experiments
```

## Troubleshooting

- `ModuleNotFoundError: No module named 'examples'` when using the CLI:
  - Use a file path for `--dataloaders-module`, e.g. `examples/loaders_template.py`, or
  - Ensure `examples` is importable (editable install and keep `examples/__init__.py`).

## License

MIT (or your preferred license)

