"""Trains a model from a yaml config. Add --resume to continue a run that got interrupted."""
import argparse

from tinygpt.trainer import load_config, train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", help="path to a yaml file in configs/")
    p.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
    p.add_argument("--max-steps", type=int, default=None, help="override max_steps from the config")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
