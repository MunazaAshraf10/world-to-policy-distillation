import argparse
import logging
from pathlib import Path

from src.training.occworld import WorldTrainer
from src.training.trainer import Trainer
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OccWorld, the teacher, or a student")
    parser.add_argument("--config", type=Path, nargs="+", required=True)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config, args.overrides)
    root = Path(__file__).resolve().parents[1]
    trainer = WorldTrainer(cfg, root) if "world" in cfg else Trainer(cfg, root)
    trainer.run()


if __name__ == "__main__":
    main()
