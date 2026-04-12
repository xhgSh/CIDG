#!/usr/bin/env bash
set -e

FUSE_VALUES="${FUSE_VALUES:-5 10 15 20}"

python main.py --config ./exps/zs_clip_cub200.json --save_zs_predictions

python main.py --config ./exps/memo_cub200.json --fuse ${FUSE_VALUES}
python main.py --config ./exps/foster_cub200.json --fuse ${FUSE_VALUES}
python main.py --config ./exps/l2p_cub200.json --fuse ${FUSE_VALUES}
python main.py --config ./exps/coda_cub200.json --fuse ${FUSE_VALUES}
python main.py --config ./exps/ease_cub200.json --fuse ${FUSE_VALUES}

