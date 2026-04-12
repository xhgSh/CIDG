#!/usr/bin/env bash
# ImageNet-A 全流程：先保存 ZS 预测，再跑各 CIL 模型并做 fuse 评估
# 数据路径：DATA_ROOT/imagenet-a/train 与 DATA_ROOT/imagenet-a/test（见 utils/data.py DATA_ROOT）
set -e

FUSE_VALUES="${FUSE_VALUES:-5 10 15 20}"

python main.py --config ./exps/zs_clip_imageneta.json --save_zs_predictions

python main.py --config ./exps/memo_imageneta.json --fuse ${FUSE_VALUES}
python main.py --config ./exps/foster_imageneta.json --fuse ${FUSE_VALUES}
python main.py --config ./exps/l2p_imageneta.json --fuse ${FUSE_VALUES}
python main.py --config ./exps/coda_imageneta.json --fuse ${FUSE_VALUES}
python main.py --config ./exps/ease_imageneta.json --fuse ${FUSE_VALUES}
