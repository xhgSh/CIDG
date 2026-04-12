import os

# 避免 libgomp 报错：若 OMP_NUM_THREADS 未设置或无效，设为 1
_val = os.environ.get("OMP_NUM_THREADS", "1").strip()
if not _val.isdigit() or int(_val) <= 0:
    os.environ["OMP_NUM_THREADS"] = "1"

# 不在这里写死 CUDA_VISIBLE_DEVICES，用哪张卡请在命令行或环境里设置，例如：
# CUDA_VISIBLE_DEVICES=0 python main.py --config=...
import json
import argparse
from trainer import train


def main():
    cli_args = setup_parser().parse_args()
    param = load_json(cli_args.config)
    args = vars(cli_args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json (defaults)
    # CLI overrides (optional)
    if cli_args.fuse is not None:
        args["fuse"] = cli_args.fuse
    if cli_args.save_zs_predictions:
        args["save_zs_predictions"] = True
    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/l2p.json',
                        help='Json file of settings.')
    parser.add_argument(
        "--save_zs_predictions",
        action="store_true",
        help="If set (only for model_name=zs_clip), save per-task zero-shot predictions to zs_result/.",
    )
    parser.add_argument(
        "--fuse",
        type=float,
        nargs="+",
        default=None,
        help="CIL+ZS logits 融合系数，可传多个值。new_logits = CIL_logits + fuse * ZS_logits。若提供则覆盖 json 配置。",
    )
    return parser


if __name__ == '__main__':
    main()