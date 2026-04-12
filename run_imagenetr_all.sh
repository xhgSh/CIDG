#!/usr/bin/env bash
set -e

FUSE_VALUES="${FUSE_VALUES:-5 10 15 20}"

python main.py --config ./exps/zs_clip_imagenetr.json --save_zs_predictions

python main.py --config ./exps/memo_imagenetr.json --fuse ${FUSE_VALUES}
python main.py --config ./exps/foster_imagenetr.json --fuse ${FUSE_VALUES}
python main.py --config ./exps/l2p_imagenetr.json --fuse ${FUSE_VALUES}
python main.py --config ./exps/coda_imagenetr.json --fuse ${FUSE_VALUES}
python main.py --config ./exps/ease_imagenetr.json --fuse ${FUSE_VALUES}

