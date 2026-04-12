#!/usr/bin/env bash
set -e

FUSE_VALUES="${FUSE_VALUES:-5 10 15 20}"

python main.py --config ./exps/zs_clip_objectnet.json --save_zs_predictions

python main.py --config ./exps/memo_objectnet.json --fuse ${FUSE_VALUES}
python main.py --config ./exps/foster_objectnet.json --fuse ${FUSE_VALUES}
python main.py --config ./exps/l2p_objectnet.json --fuse ${FUSE_VALUES}
python main.py --config ./exps/coda_objectnet.json --fuse ${FUSE_VALUES}
python main.py --config ./exps/ease_objectnet.json --fuse ${FUSE_VALUES}

