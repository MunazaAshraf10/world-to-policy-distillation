from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    from src.verification.config import load_config
    from src.verification.report import write_report
    from src.verification.runner import preflight, run_verification, seed_run

    parser = argparse.ArgumentParser(
        description="Verify frozen OccWorld under its official conditioning protocol"
    )
    parser.add_argument("--config", type=Path, default=project_root / "configs/debug.yaml")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--infos", type=Path)
    parser.add_argument("--sample-index", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    overrides = {
        key: getattr(args, key)
        for key in ("checkpoint", "data_root", "infos", "sample_index", "device", "output")
    }
    config = load_config(args.config, overrides, project_root)
    if args.preflight_only and args.output is None:
        parser.error("--preflight-only requires --output to keep its report separate from Stage 1")
    if config.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {config.output}; choose a new path or pass --overwrite"
        )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger("verification")
    if args.preflight_only:
        seed_run(config.seed)
        report = preflight(config)
        report["config"] = config.resolved()
    else:
        logger.info(
            "Running official occupancy verification; future gt_mode conditioning is retained"
        )
        report = run_verification(config, project_root)
    write_report(config.output, report, overwrite=args.overwrite)
    logger.info("%s: %s", report["status"], config.output)


if __name__ == "__main__":
    main()
