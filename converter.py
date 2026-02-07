import argparse
import logging
import sys
from pathlib import Path

import tyro

from lerobot_hdf5.config import LeRobotConfig
from lerobot_hdf5.core import LeRobotDatasetConverter

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("lerobot_hdf5")


def main():
    # 1. Pre-parse just the --config flag to set defaults
    # We use standard argparse here just to peek at --config safely
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=None)

    # parse_known_args allows us to grab --config if it exists,
    # and ignore the other flags (repo-id, input-path, etc.) for now.
    args, remaining_argv = parser.parse_known_args()

    # 2. Load defaults if config file is provided
    default_cfg = None
    if args.config:
        if not args.config.exists():
            logger.error(f"Config file not found: {args.config}")
            sys.exit(1)

        logger.info(f"Loading defaults from {args.config}")
        try:
            default_cfg = LeRobotConfig.from_yaml(args.config)
        except Exception as e:
            logger.error(f"Error parsing YAML: {e}")
            sys.exit(1)

    # 3. Use Tyro to parse the full command line
    # Tyro will layer CLI args (remaining_argv) ON TOP of default_cfg.
    # We don't need to filter sys.argv manually anymore because
    # we aren't "removing" flags, we are just using tyro to resolve the final object.

    try:
        # Note: We pass the FULL sys.argv[1:] to tyro.
        # Tyro will see --config, but since LeRobotConfig doesn't have a 'config' field,
        # it might complain unless we handle it.
        #
        # TRICK: To make tyro ignore the --config flag we already handled,
        # we pass the 'remaining_argv' which argparse filtered out for us?
        # NO, argparse removed --config but kept the values.
        #
        # BETTER APPROACH:
        # Just use the 'default' parameter of tyro.cli.
        # If the user passes --config, we load that object and pass it as `default`.
        # But we must ensure --config isn't passed to tyro if it's not in the dataclass.

        # Filter --config out of the args passed to tyro so it doesn't error
        # saying "unrecognized argument: --config"
        tyro_args = [
            arg
            for arg in sys.argv[1:]
            if arg != "--config" and str(args.config) not in arg
        ]

        cfg: LeRobotConfig | None = tyro.cli(
            LeRobotConfig, default=default_cfg, args=tyro_args
        )

    except SystemExit:
        # Tyro handles --help automatically, so we just exit gracefully
        sys.exit(0)
    except Exception as e:
        logger.error(f"Configuration Error: {e}")
        logger.error(
            "Usage: python convert.py --repo-id <id> --input-path <path> [--config <file.yaml>]"
        )
        sys.exit(1)

    # 4. Run Conversion
    if not isinstance(cfg, LeRobotConfig):
        logger.error(
            "Failed to parse configuration. Please check your command line arguments and config file."
        )
        sys.exit(1)
    else:
        try:
            converter = LeRobotDatasetConverter(cfg)
            converter.run_conversion()
        except Exception:
            logger.exception("Fatal error during conversion")
            sys.exit(1)


if __name__ == "__main__":
    main()
