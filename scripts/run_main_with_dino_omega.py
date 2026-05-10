import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import setup_cfg
from utils.util import set_gpu, set_seed
from models.bimc_dino_fusion import BiMCDinoFusion


def main():
    parser = argparse.ArgumentParser(description="Run main.py with a runtime Dino/CLIP visual omega override")
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--dino_omega", type=float, required=True)
    args = parser.parse_args()

    BiMCDinoFusion.OMEGA = float(args.dino_omega)
    cfg = setup_cfg(args.data_cfg, args.train_cfg)
    set_seed(cfg.SEED)
    set_gpu(cfg.DEVICE.GPU_ID)

    from engine.engine import Runner

    engine = Runner(cfg)
    engine.run()


if __name__ == "__main__":
    main()
