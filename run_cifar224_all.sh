#!/usr/bin/env bash
set -e

FUSE_VALUES="${FUSE_VALUES:-5 10 15 20}"

python main.py --config ./exps/zs_clip_cifar224.json --save_zs_predictions

python main.py --config ./exps/memo_cifar224.json --fuse ${FUSE_VALUES}
python main.py --config ./exps/foster_cifar224.json --fuse ${FUSE_VALUES}
python main.py --config ./exps/l2p_cifar224.json --fuse ${FUSE_VALUES}
python main.py --config ./exps/coda_cifar224.json --fuse ${FUSE_VALUES}
python main.py --config ./exps/ease_cifar224.json --fuse ${FUSE_VALUES}

