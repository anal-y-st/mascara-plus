
import logging
from pathlib import Path
def setup_logging(workdir: str):
    Path(workdir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mascara"); logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(Path(workdir)/"train.log"); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger
