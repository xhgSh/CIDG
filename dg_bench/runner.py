"""
CIDG (Class-Incremental Domain Generalization) runner for C3Box.
Leave-one-domain-out: one domain as test, rest as source; source split 80% train / 20% val.
Reuses C3Box CIL models; evaluation is on the held-out test domain only.
Logs to ./log/{timestamp}_{experiment_name}/ with args, scores, and forgetting curve.
"""
import os
import sys

# 避免 libgomp: Invalid value for environment variable OMP_NUM_THREADS
if os.environ.get("OMP_NUM_THREADS", "").strip() in ("", "0") or not os.environ.get("OMP_NUM_THREADS", "").strip().isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"
import json
import argparse
import logging
from copy import deepcopy
from typing import List, Dict, Any, Optional
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

# Ensure C3Box root is on path when running as script
_C3BOX_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _C3BOX_ROOT not in sys.path:
    sys.path.insert(0, _C3BOX_ROOT)

from utils import factory
from dg_bench.domainbed_dm import DomainBedDataManager


# C3Box model names supported by factory (exclude coop, l2p_without which have no impl)
C3BOX_MODEL_NAMES = [
    "proof", "simplecil", "zs_clip", "rapf", "coda", "dualprompt", "l2p", "memo", "foster",
    "clg_cbm", "mind", "bofa", "engine", "finetune", "ease", "tuna",
    "aper_adapter", "aper_ssf", "aper_vpt", "aper_finetune",
]

# Map model_name -> exps json filename (without .json)
EXPS_MAP = {
    "dualprompt": "dual",
    "aper_vpt": None,  # set by vpt_type: aper_vpt_deep / aper_vpt_shallow
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="CIDG: Leave-one-domain-out CIL on DomainBed datasets (C3Box)"
    )
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Root dir that contains all DG datasets (e.g. .../PACS, .../VLCS)")
    parser.add_argument("--dataset_name", type=str, required=True,
                        choices=["PACS", "VLCS", "OfficeHome", "terra_incognita", "domain_net"])
    parser.add_argument("--val_domain", type=str, default=None,
                        help="Domain to use as test. If omitted, iterate all domains (leave-one-out).")
    parser.add_argument("--convnet_type", type=str, default="clip",
                        help="Backbone type (C3Box uses clip; kept for CLI compatibility)")
    parser.add_argument("--model_name", type=str, required=True,
                        choices=C3BOX_MODEL_NAMES)
    parser.add_argument("--device", type=int, nargs="+", default=[0])
    parser.add_argument("--seed", type=int, nargs="+", default=[1],
                        help="Random seeds to run (default: [1] for single run; use e.g. --seed 0 1 2 for DomainBed 3-seed protocol)")
    parser.add_argument("--init_cls", type=int, default=0)
    parser.add_argument("--increment", type=int, default=10)
    parser.add_argument("--memory_size", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--init_lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--min_lr", type=float, default=1e-8)
    parser.add_argument("--tuned_epoch", type=int, default=10)
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--no-shuffle", action="store_false", dest="shuffle")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--val_split_ratio", type=float, default=0.2)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--use_validation_for_selection", action="store_true", default=False,
                        help="Use source 20%% val for checkpoint selection (not implemented for all C3Box models)")
    parser.add_argument("--no-use_validation_for_selection", action="store_false", dest="use_validation_for_selection")
    parser.add_argument("--temperature_scaling", action="store_true", default=False)
    parser.add_argument("--no-temperature_scaling", action="store_false", dest="temperature_scaling")
    parser.add_argument("--temperature_candidates", type=float, nargs="+", default=[0.8, 1.0, 1.2])
    parser.add_argument("--sigma_scaling", action="store_true", default=False)
    parser.add_argument("--no-sigma_scaling", action="store_false", dest="sigma_scaling")
    parser.add_argument("--sigma_candidates", type=float, nargs="+", default=[0.8, 1.0, 1.2])
    parser.add_argument("--prefix", type=str, default="cidg")
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--save_cmd", action="store_true")
    parser.add_argument("--allow_pretrain_skip", action="store_true")
    parser.add_argument("--hf_offline", action="store_true")
    parser.add_argument("--hf_home", type=str, default=None)
    parser.add_argument("--vpt_type", type=str, default="deep", choices=["shallow", "deep"])
    parser.add_argument("--prompt_token_num", type=int, default=None)
    parser.add_argument("--ffn_num", type=int, default=None)
    # Image-level augmentation (train expansion + test-time augmentation)
    parser.add_argument("--aug_enable", action="store_true", help="Enable image augmentation (train expansion + TTA)")
    parser.add_argument("--aug_magnitude", type=float, default=3.0, help="Augmentation severity (1-10 scale)")
    parser.add_argument("--aug_methods", type=str, default="all",
                        choices=["style", "shape", "deformation", "style_shape", "style_deformation", "shape_deformation", "all"],
                        help="Train preset: style / shape / deformation / style_shape / style_deformation / shape_deformation / all")
    parser.add_argument("--aug_tta_methods", type=str, default="all",
                        choices=["style", "shape", "deformation", "style_shape", "style_deformation", "shape_deformation", "all"],
                        help="TTA preset: same as aug_methods for consistent chain")
    parser.add_argument("--aug_train_views", type=int, default=0, help="Extra augmented views per train sample (0=off)")
    parser.add_argument("--aug_test_views", type=int, default=0, help="TTA views at test (0=off)")
    parser.add_argument("--aug_chain_depth", type=int, default=-1, help="AugMix chain depth (-1=random 1..3)")
    parser.add_argument("--aug_chain_width", type=int, default=1)
    parser.add_argument("--aug_do_elastic", action="store_true", default=True)
    parser.add_argument("--no-aug_do_elastic", action="store_false", dest="aug_do_elastic")
    parser.add_argument("--aug_do_grid", action="store_true", default=True)
    parser.add_argument("--no-aug_do_grid", action="store_false", dest="aug_do_grid")
    parser.add_argument("--aug_include_flip", action="store_true", default=True)
    parser.add_argument("--no-aug_include_flip", action="store_false", dest="aug_include_flip")
    parser.add_argument("--aug_max_views_cap", type=int, default=16)
    parser.add_argument("--aug_use_consistent_chain", action="store_true", default=True,
                        help="Train and TTA use same full chain (AugMix 13 + hflip + elastic + grid)")
    parser.add_argument("--no-aug_use_consistent_chain", action="store_false", dest="aug_use_consistent_chain")
    # CIL + ZS logits 融合：从 zs_result 加载 zs_clip 预存 logits；new_logits = CIL_logits + fuse * ZS_logits。
    parser.add_argument(
        "--fuse",
        type=float,
        nargs="+",
        default=None,
        help="融合系数，可传多个值。new_logits = CIL_logits + fuse * ZS_logits。未给出则不融合。",
    )
    parser.add_argument(
        "--save_zs_predictions",
        action="store_true",
        help="Run with model zs_clip and save per-task CLIP zero-shot predictions (with logits) to zs_result/ for fusion.",
    )
    return parser.parse_args()


def _dataset_dir(dataset_root: str, dataset_name: str) -> str:
    from dg_bench.domainbed_dm import DATASET_DIR_MAP
    folder = DATASET_DIR_MAP.get(dataset_name, dataset_name)
    return os.path.join(dataset_root, folder)


def discover_domains(dataset_root: str, dataset_name: str) -> List[str]:
    base = _dataset_dir(dataset_root, dataset_name)
    if not os.path.isdir(base):
        raise FileNotFoundError("Dataset not found: {}".format(base))
    domains = [
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and not d.startswith(".")
    ]
    if not domains:
        raise RuntimeError("No domains under {}".format(base))
    return sorted(domains)


def load_default_args(model_name: str, vpt_type: Optional[str] = None) -> Dict[str, Any]:
    """Load default args from exps/{name}.json for the given model."""
    name = model_name.lower()
    exp_file = EXPS_MAP.get(name)
    if exp_file is None and name == "aper_vpt":
        exp_file = "aper_vpt_deep" if (vpt_type or "deep") == "deep" else "aper_vpt_shallow"
    elif exp_file is None:
        exp_file = name
    path = os.path.join(_C3BOX_ROOT, "exps", exp_file + ".json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_args_dict(cli_args) -> Dict[str, Any]:
    """Merge exps default + CLI into args dict for C3Box factory and learners."""
    defaults = load_default_args(cli_args.model_name, getattr(cli_args, "vpt_type", None))
    args_dict = deepcopy(defaults)
    # CLI overrides
    args_dict["dataset_root"] = cli_args.dataset_root
    args_dict["dataset_name"] = cli_args.dataset_name
    args_dict["val_domain"] = cli_args.val_domain
    args_dict["model_name"] = cli_args.model_name
    args_dict["device"] = list(cli_args.device)
    args_dict["seed"] = list(cli_args.seed)
    args_dict["init_cls"] = cli_args.init_cls
    args_dict["increment"] = cli_args.increment
    args_dict["memory_size"] = cli_args.memory_size
    args_dict["batch_size"] = cli_args.batch_size
    args_dict["init_lr"] = cli_args.init_lr
    args_dict["weight_decay"] = cli_args.weight_decay
    args_dict["min_lr"] = cli_args.min_lr
    args_dict["tuned_epoch"] = cli_args.tuned_epoch
    # bofa OLF Stage 2 在 CIDG 下易导致 Train_acc 归零、测试崩盘，默认关闭 Stage 2 训练（仅用 Stage 1 + 结束时 prepare_stage2 存权重）
    if cli_args.model_name.lower() == "bofa":
        args_dict["epoch"] = 0
    args_dict["shuffle"] = cli_args.shuffle
    args_dict["image_size"] = cli_args.image_size
    args_dict["val_split_ratio"] = cli_args.val_split_ratio
    args_dict["prefix"] = cli_args.prefix
    args_dict["dataset"] = cli_args.dataset_name  # C3Box compatibility
    if cli_args.prompt_token_num is not None:
        args_dict["prompt_token_num"] = cli_args.prompt_token_num
    if cli_args.ffn_num is not None:
        args_dict["ffn_num"] = cli_args.ffn_num
    args_dict["vpt_type"] = getattr(cli_args, "vpt_type", "deep")
    # Image augmentation
    args_dict["aug_enable"] = getattr(cli_args, "aug_enable", False)
    args_dict["aug_magnitude"] = getattr(cli_args, "aug_magnitude", 3.0)
    args_dict["aug_methods"] = getattr(cli_args, "aug_methods", "all")
    args_dict["aug_tta_methods"] = getattr(cli_args, "aug_tta_methods", "all")
    args_dict["aug_train_views"] = getattr(cli_args, "aug_train_views", 0)
    args_dict["aug_test_views"] = getattr(cli_args, "aug_test_views", 0)
    args_dict["aug_chain_depth"] = getattr(cli_args, "aug_chain_depth", -1)
    args_dict["aug_chain_width"] = getattr(cli_args, "aug_chain_width", 1)
    args_dict["aug_do_elastic"] = getattr(cli_args, "aug_do_elastic", True)
    args_dict["aug_do_grid"] = getattr(cli_args, "aug_do_grid", True)
    args_dict["aug_include_flip"] = getattr(cli_args, "aug_include_flip", True)
    args_dict["aug_max_views_cap"] = getattr(cli_args, "aug_max_views_cap", 16)
    args_dict["aug_use_consistent_chain"] = getattr(cli_args, "aug_use_consistent_chain", True)
    args_dict["fuse"] = getattr(cli_args, "fuse", None)
    args_dict["save_zs_predictions"] = getattr(cli_args, "save_zs_predictions", False)
    # Ensure backbone for CLIP
    if "backbone_type" not in args_dict or not args_dict["backbone_type"]:
        args_dict["backbone_type"] = "clip"
    # Device as list of torch.device
    dev_list = args_dict["device"]
    if dev_list and dev_list != [-1]:
        args_dict["device"] = [torch.device("cuda:{}".format(d)) for d in dev_list]
    else:
        args_dict["device"] = [torch.device("cpu")]
    return args_dict


def setup_logger(log_dir: str, log_name: str):
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, log_name + ".log")
    kwargs = dict(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(filename=logfile, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    if sys.version_info >= (3, 8):
        kwargs["force"] = True  # 确保重配 root logger，避免被其他库先调 basicConfig 导致本进程写不进文件
    logging.basicConfig(**kwargs)


def _set_random(seed: int = 1):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_once(args_dict: Dict[str, Any], train_domains: List[str], test_domain: str, seed: int) -> Dict[str, Any]:
    if args_dict.get("hf_offline"):
        os.environ["HF_HUB_OFFLINE"] = "1"
    if args_dict.get("hf_home"):
        os.environ["HF_HOME"] = str(args_dict["hf_home"])

    dm = DomainBedDataManager(
        dataset_root=args_dict["dataset_root"],
        dataset_name=args_dict["dataset_name"],
        train_domains=train_domains,
        test_domain=test_domain,
        shuffle=args_dict["shuffle"],
        seed=seed,
        init_cls=args_dict["init_cls"],
        increment=args_dict["increment"],
        image_size=args_dict.get("image_size", 224),
        val_split_ratio=args_dict.get("val_split_ratio", 0.2),
    )

    _set_random(seed)
    # 使用当前数据集的总类数，避免 L2P 等模型用 exps 里固定 nb_classes 导致 task>0 时标签越界 (t >= n_classes)
    args_dict["nb_classes"] = dm.get_total_classnum()
    from utils.image_aug import ImageAugHelper
    aug_helper = ImageAugHelper(args_dict)
    if aug_helper.enabled:
        logging.info("Image aug: train_views=%s test_views=%s methods=%s tta_methods=%s magnitude=%s",
                     aug_helper.num_train_views(), aug_helper.num_test_views(),
                     args_dict.get("aug_methods", "all"), args_dict.get("aug_tta_methods", "all"),
                     args_dict.get("aug_magnitude", 3.0))
    try:
        learner = factory.get_model(args_dict["model_name"], args_dict)
    except Exception as e:
        if args_dict.get("allow_pretrain_skip") and ("safetensors" in str(e) or "load" in str(e).lower() or "timeout" in str(e).lower()):
            logging.warning("Pretrained load failed, continuing with --allow_pretrain_skip: %s", e)
            learner = factory.get_model(args_dict["model_name"], args_dict)
        else:
            raise

    learner.data_manager = dm
    dm.aug_helper = aug_helper
    learner.aug_helper = aug_helper
    if hasattr(learner, "save_dir"):
        learner.save_dir = args_dict.get("save_dir", "")
    try:
        learner.fuse = args_dict.get("fuse", None)
    except Exception:
        pass

    save_zs_predictions = args_dict.get("save_zs_predictions", False)
    _fuse = args_dict.get("fuse", None)
    if isinstance(_fuse, (list, tuple)):
        _fuse_list = [float(x) for x in _fuse]
    elif _fuse is None:
        _fuse_list = []
    else:
        _fuse_list = [float(_fuse)]
    _fuse_list = sorted(set(_fuse_list))
    use_fuse = len(_fuse_list) > 0
    is_zs_clip = (args_dict.get("model_name") or "").lower() == "zs_clip"
    if save_zs_predictions and not is_zs_clip:
        logging.warning("--save_zs_predictions is ignored when model_name is not zs_clip.")
        save_zs_predictions = False
    if save_zs_predictions:
        logging.info("Save zs predictions: ON -> writing to zs_result/%s/%s_seed%s/task_*.csv",
                     args_dict["dataset_name"], test_domain, seed)

    # zs_result 目录：按 数据集 / 测试域_seed / task_{id}.csv 存放，与增量阶段对齐
    zs_result_dir = os.path.join(_C3BOX_ROOT, "zs_result", args_dict["dataset_name"], "{}_seed{}".format(test_domain, seed))
    if use_fuse:
        logging.info("Fuse ON: new_logits = CIL_logits + fuse * ZS_logits | fuse = %s | load dir: %s",
                     ",".join("%.4f" % x for x in _fuse_list), zs_result_dir)

    logging.info("Dataset: %s | Train domains: %s | Test domain: %s | Seed: %s",
                  args_dict["dataset_name"], ",".join(train_domains), test_domain, seed)
    logging.info("Classes: %s | Tasks: %s | Increment: %s", dm.get_total_classnum(), dm.nb_tasks, args_dict["increment"])

    vit_curve = {"top1": []}
    vit_curve_fuse: Dict[float, Dict[str, List[float]]] = {f: {"top1": []} for f in _fuse_list}
    nme_curve = {"top1": []}

    for task in tqdm(range(dm.nb_tasks), desc="Tasks", ncols=100):
        learner.incremental_train(dm)

        # 保存 zs_clip 当前阶段的预测到 zs_result，供后续门控加载（每阶段类别空间不同，按阶段存）
        if save_zs_predictions and is_zs_clip and hasattr(learner, "get_zs_predictions_and_logits"):
            try:
                y_pred_zs, y_true_zs, logits = learner.get_zs_predictions_and_logits()
                os.makedirs(zs_result_dir, exist_ok=True)
                csv_path = os.path.join(zs_result_dir, "task_{}.csv".format(task))
                import csv as csv_module
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    w = csv_module.writer(f)
                    C = logits.shape[1]
                    w.writerow(["y_true", "y_pred_zs"] + ["logit_{}".format(j) for j in range(C)])
                    for i in range(len(y_true_zs)):
                        w.writerow([int(y_true_zs[i]), int(y_pred_zs[i, 0])] + [float(logits[i, j]) for j in range(C)])
                logging.info("Saved zs_clip predictions to %s", csv_path)
            except Exception as e:
                logging.warning("Failed to save zs_clip predictions for task %s: %s", task, e)

        # 融合：从 zs_result 加载当前 task 的 ZS logits 表
        if use_fuse and not is_zs_clip:
            learner.zs_gate_table = None
            try:
                import pandas as pd
                csv_path = os.path.join(zs_result_dir, "task_{}.csv".format(task))
                if os.path.isfile(csv_path):
                    learner.zs_gate_table = pd.read_csv(csv_path)
                else:
                    logging.warning("fuse 已设置但缺少 %s; fusion disabled for this task.", csv_path)
            except Exception as e:
                logging.warning("Failed to load zs_result for fusion: %s", e)

        result = learner.eval_task()
        if isinstance(result, (list, tuple)):
            cnn_accy = result[0]
            nme_accy = result[1] if len(result) > 1 else None
            fuse_accy = result[2] if len(result) > 2 else None
        else:
            cnn_accy, nme_accy = result, None
            fuse_accy = None
        learner.after_task()

        vit_curve["top1"].append(cnn_accy["top1"])
        fuse_accy_norm = None
        if fuse_accy is not None and isinstance(fuse_accy, dict):
            fuse_accy_norm = {}
            for k, v in fuse_accy.items():
                try:
                    kk = float(k)
                except Exception:
                    continue
                if isinstance(v, dict) and "top1" in v:
                    fuse_accy_norm[kk] = v
            for f in _fuse_list:
                if f in fuse_accy_norm:
                    vit_curve_fuse[f]["top1"].append(fuse_accy_norm[f]["top1"])
        if nme_accy is not None:
            nme_curve["top1"].append(nme_accy["top1"])
        num_cls = getattr(learner, "_total_classes", None)
        if num_cls is not None:
            logging.info(
                "Task %d/%d | Test (held-out) top1 (融合前 CIL): %.2f%% (eval on %d classes)",
                task + 1,
                dm.nb_tasks,
                cnn_accy["top1"],
                num_cls,
            )
        else:
            logging.info(
                "Task %d/%d | Test (held-out) top1 (融合前 CIL): %.2f%%",
                task + 1,
                dm.nb_tasks,
                cnn_accy["top1"],
            )
        if fuse_accy_norm is not None:
            for f in _fuse_list:
                if f in fuse_accy_norm and isinstance(fuse_accy_norm[f], dict) and "top1" in fuse_accy_norm[f]:
                    logging.info(
                        "Task %d/%d | Test (held-out) top1 (融合后 fuse=%.4f): %.2f%%",
                        task + 1,
                        dm.nb_tasks,
                        float(f),
                        fuse_accy_norm[f]["top1"],
                    )
        if nme_accy is not None:
            logging.info("  NME top1: %.2f%%", nme_accy["top1"])
        print(
            "  [融合前] Task %d/%d 测试域 top1 (CIL): %.2f%%"
            % (task + 1, dm.nb_tasks, cnn_accy["top1"]),
            flush=True,
        )
        if fuse_accy_norm is not None:
            for f in _fuse_list:
                if f in fuse_accy_norm and isinstance(fuse_accy_norm[f], dict) and "top1" in fuse_accy_norm[f]:
                    print(
                        "  [融合后] Task %d/%d fuse=%.4f: %.2f%% (融合前 %.2f%%)"
                        % (task + 1, dm.nb_tasks, float(f), fuse_accy_norm[f]["top1"], cnn_accy["top1"]),
                        flush=True,
                    )

    # 评估指标: 平均准确率 Avg Acc (Ā) = 所有增量阶段平均; 最终准确率 Last Acc (A_B) = 最后一阶段后性能
    last_acc = vit_curve["top1"][-1] if vit_curve["top1"] else 0.0
    avg_acc = (sum(vit_curve["top1"]) / len(vit_curve["top1"])) if vit_curve["top1"] else 0.0
    avg_last_fuse_by_fuse: Dict[float, Dict[str, float]] = {}
    for f in _fuse_list:
        lst = vit_curve_fuse.get(f, {}).get("top1", [])
        if lst:
            avg_last_fuse_by_fuse[float(f)] = {
                "avg": float(sum(lst) / len(lst)),
                "last": float(lst[-1]),
            }
    final_nme = (sum(nme_curve["top1"]) / len(nme_curve["top1"])) if nme_curve["top1"] else None
    logging.info("Avg Acc (Ā) CIL (融合前): %.2f%% | Last Acc (A_B) CIL: %.2f%%", avg_acc, last_acc)
    if avg_last_fuse_by_fuse:
        for f in _fuse_list:
            if float(f) in avg_last_fuse_by_fuse:
                logging.info(
                    "Avg/Last Acc (Ā/A_B) 融合后 fuse=%.4f: %.2f%% / %.2f%%",
                    float(f),
                    avg_last_fuse_by_fuse[float(f)]["avg"],
                    avg_last_fuse_by_fuse[float(f)]["last"],
                )
    print("", flush=True)
    print("======== 本域实验得分 (Test domain: %s, Seed: %s) ========" % (test_domain, seed), flush=True)
    print("  各阶段测试域 top1 (融合前 CIL): %s" % (", ".join("%.2f%%" % x for x in vit_curve["top1"])), flush=True)
    print("  平均准确率 Avg Acc (Ā) CIL: %.2f%%" % avg_acc, flush=True)
    print("  最终准确率 Last Acc (A_B) CIL: %.2f%%" % last_acc, flush=True)
    if avg_last_fuse_by_fuse:
        for f in _fuse_list:
            ff = float(f)
            if ff in vit_curve_fuse and vit_curve_fuse[ff]["top1"]:
                print(
                    "  各阶段测试域 top1 (融合后 fuse=%.4f): %s"
                    % (ff, ", ".join("%.2f%%" % x for x in vit_curve_fuse[ff]["top1"])),
                    flush=True,
                )
                print(
                    "  平均准确率 Avg Acc (Ā) 融合 fuse=%.4f: %.2f%%"
                    % (ff, avg_last_fuse_by_fuse[ff]["avg"]),
                    flush=True,
                )
                print(
                    "  最终准确率 Last Acc (A_B) 融合 fuse=%.4f: %.2f%%"
                    % (ff, avg_last_fuse_by_fuse[ff]["last"]),
                    flush=True,
                )
    print("==========================================================", flush=True)

    if len(_fuse_list) == 1 and float(_fuse_list[0]) in avg_last_fuse_by_fuse:
        _f0 = float(_fuse_list[0])
        avg_acc_fuse = avg_last_fuse_by_fuse[_f0]["avg"]
        last_acc_fuse = avg_last_fuse_by_fuse[_f0]["last"]
        task_top1_list_fuse = vit_curve_fuse[_f0]["top1"] if vit_curve_fuse[_f0]["top1"] else None
    else:
        avg_acc_fuse = None
        last_acc_fuse = None
        task_top1_list_fuse = None

    return {
        "val_domain": test_domain,
        "train_domains": train_domains,
        "seed": seed,
        "classes": dm.get_total_classnum(),
        "tasks": dm.nb_tasks,
        "increment": args_dict["increment"],
        "avg_acc": avg_acc,
        "last_acc": last_acc,
        "final_task_top1": last_acc,
        "task_top1_list": vit_curve["top1"],
        "avg_acc_fuse": avg_acc_fuse,
        "last_acc_fuse": last_acc_fuse,
        "task_top1_list_fuse": task_top1_list_fuse,
        "fuse_list": _fuse_list if _fuse_list else None,
        "avg_last_fuse_by_fuse": avg_last_fuse_by_fuse if avg_last_fuse_by_fuse else None,
        "task_top1_list_fuse_by_fuse": {f: vit_curve_fuse[f]["top1"] for f in vit_curve_fuse if vit_curve_fuse[f]["top1"]}
        if any(vit_curve_fuse[f]["top1"] for f in vit_curve_fuse) else None,
        "final_nme_mean": final_nme,
    }


def plot_forgetting_curve(task_top1_list: List[float], save_path: str, title: str = "CIDG Forgetting Curve"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        logging.warning("matplotlib not available; skipping forgetting curve plot.")
        return
    tasks = list(range(1, len(task_top1_list) + 1))
    plt.figure(figsize=(6, 4))
    plt.plot(tasks, task_top1_list, marker="o", linewidth=2, markersize=6)
    plt.xlabel("Task (after learning up to this task)")
    plt.ylabel("Accuracy on test domain (%)")
    plt.title(title)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_all_domains_curve(
    all_results: List[Dict[str, Any]],
    eval_plan: List[str],
    dataset_name: str,
    model_name: str,
    save_path: str,
):
    """Plot one accuracy curve per test domain (color-coded), legend upper right, title = dataset + method."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        logging.warning("matplotlib not available; skipping all-domains curve plot.")
        return
    # Group by val_domain, then average task_top1_list over seeds per domain
    from collections import defaultdict
    by_domain = defaultdict(list)
    for r in all_results:
        by_domain[r["val_domain"]].append(r["task_top1_list"])
    # Order by eval_plan
    domain_order = [d for d in eval_plan if d in by_domain]
    if not domain_order:
        return
    n_tasks = len(by_domain[domain_order[0]][0])
    plt.figure(figsize=(7, 5))
    colors = plt.cm.tab10.colors if len(domain_order) <= 10 else plt.cm.tab20.colors
    for i, vd in enumerate(domain_order):
        lists = by_domain[vd]
        avg_list = [sum(lists[s][t] for s in range(len(lists))) / len(lists) for t in range(n_tasks)]
        tasks = list(range(1, n_tasks + 1))
        plt.plot(tasks, avg_list, marker="o", linewidth=2, markersize=5, color=colors[i % len(colors)], label=vd)
    plt.xlabel("Task (after learning up to this task)")
    plt.ylabel("Accuracy on test domain (%)")
    plt.title("{}  {}".format(dataset_name, model_name))
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    cli = parse_args()
    domains = discover_domains(cli.dataset_root, cli.dataset_name)
    if cli.val_domain is not None:
        if cli.val_domain not in domains:
            raise ValueError("val_domain {} not in {}".format(cli.val_domain, domains))
        eval_plan = [cli.val_domain]
    else:
        eval_plan = domains

    args_dict = build_args_dict(cli)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = cli.run_tag or "cidg_{}_{}".format(cli.dataset_name, cli.model_name)
    log_dir = os.path.join(_C3BOX_ROOT, "log", "{}_{}".format(timestamp, exp_name))
    os.makedirs(log_dir, exist_ok=True)
    setup_logger(log_dir, "runner")
    args_dict["save_dir"] = log_dir

    all_results = []
    for vd in eval_plan:
        train_domains = [d for d in domains if d != vd]
        logging.info("===== Test domain: %s | Train domains: %s =====", vd, ",".join(train_domains))
        for seed in cli.seed:
            logging.info("--- Seed: %s ---", seed)
            res = run_once(args_dict, train_domains, vd, seed)
            all_results.append(res)
            # Per-run forgetting curve
            curve_path = os.path.join(log_dir, "forgetting_curve_{}_seed{}.png".format(vd, seed))
            plot_forgetting_curve(
                res["task_top1_list"],
                curve_path,
                title="CIDG {} {} (test={}, seed={})".format(cli.dataset_name, cli.model_name, vd, seed),
            )

    # Average forgetting curve (average across seeds per task index)
    if all_results and len(eval_plan) == 1 and len(cli.seed) > 1:
        n_tasks = len(all_results[0]["task_top1_list"])
        avg_list = [
            sum(r["task_top1_list"][t] for r in all_results) / len(all_results)
            for t in range(n_tasks)
        ]
        plot_forgetting_curve(
            avg_list,
            os.path.join(log_dir, "forgetting_curve_avg.png"),
            title="CIDG {} {} (test={}, avg over seeds)".format(cli.dataset_name, cli.model_name, eval_plan[0]),
        )

    # All-domains curve: one line per test domain (color-coded), legend upper right, title = dataset + method
    if all_results:
        plot_all_domains_curve(
            all_results,
            eval_plan,
            cli.dataset_name,
            cli.model_name,
            os.path.join(log_dir, "forgetting_curve_all_domains.png"),
        )

    # Save args and results
    args_save = deepcopy(args_dict)
    if "device" in args_save and isinstance(args_save["device"], list) and args_save["device"]:
        args_save["device"] = [str(d) for d in args_save["device"]]
    with open(os.path.join(log_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(args_save, f, indent=2, ensure_ascii=False)

    with open(os.path.join(log_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    if cli.save_cmd:
        with open(os.path.join(log_dir, "command.txt"), "w", encoding="utf-8") as f:
            f.write(" ".join(sys.argv) + "\n")

    # Summary log
    with open(os.path.join(log_dir, "experiment_log.txt"), "w", encoding="utf-8") as f:
        f.write("=== CIDG Experiment Log ===\n")
        f.write("Dataset: {} | Model: {}\n".format(cli.dataset_name, cli.model_name))
        f.write("Log dir: {}\n".format(log_dir))
        f.write("Command: {}\n".format(" ".join(sys.argv)))
        f.write("=" * 50 + "\n\n")
        for r in all_results:
            f.write(
                "Test domain: {} | Seed: {} | Avg Acc (Ā) CIL: {:.2f}% | Last Acc (A_B) CIL: {:.2f}%".format(
                    r["val_domain"], r["seed"], r["avg_acc"], r["last_acc"]
                )
            )
            if r.get("avg_acc_fuse") is not None and r.get("last_acc_fuse") is not None:
                f.write(
                    " | Avg Acc (Ā) 融合: {:.2f}% | Last Acc (A_B) 融合: {:.2f}%".format(
                        r["avg_acc_fuse"], r["last_acc_fuse"]
                    )
                )
            if r.get("avg_last_fuse_by_fuse"):
                parts = []
                for fv, d in sorted(r["avg_last_fuse_by_fuse"].items(), key=lambda kv: float(kv[0])):
                    parts.append("fuse=%.4f: avg=%.2f last=%.2f" % (float(fv), float(d.get("avg", 0.0)), float(d.get("last", 0.0))))
                if parts:
                    f.write(" | 融合 by fuse: " + " | ".join(parts))
            f.write("\n")
            f.write(
                "  Task top1 list (CIL): {}\n".format(
                    ", ".join("{:.2f}".format(x) for x in r["task_top1_list"])
                )
            )
            if r.get("task_top1_list_fuse"):
                f.write(
                    "  Task top1 list (融合): {}\n".format(
                        ", ".join("{:.2f}".format(x) for x in r["task_top1_list_fuse"])
                    )
                )
            if r.get("task_top1_list_fuse_by_fuse"):
                for fv, lst in sorted(r["task_top1_list_fuse_by_fuse"].items(), key=lambda kv: float(kv[0])):
                    if lst:
                        f.write(
                            "  Task top1 list (融合 fuse=%.4f): %s\n" % (
                                float(fv), ", ".join("{:.2f}".format(x) for x in lst)
                            )
                        )
        if all_results:
            mean_avg = sum(r["avg_acc"] for r in all_results) / len(all_results)
            mean_last = sum(r["last_acc"] for r in all_results) / len(all_results)
            fuse_results = [r for r in all_results if r.get("avg_acc_fuse") is not None]
            if fuse_results:
                mean_avg_fuse = sum(r["avg_acc_fuse"] for r in fuse_results) / len(fuse_results)
                mean_last_fuse = sum(r["last_acc_fuse"] for r in fuse_results) / len(fuse_results)
                f.write(
                    "\nOverall mean Avg Acc (Ā) CIL: {:.2f}% | mean Last Acc (A_B) CIL: {:.2f}%\n".format(
                        mean_avg, mean_last
                    )
                )
                f.write(
                    "Overall mean Avg Acc (Ā) 融合: {:.2f}% | mean Last Acc (A_B) 融合: {:.2f}%\n".format(
                        mean_avg_fuse, mean_last_fuse
                    )
                )
            else:
                f.write(
                    "\nOverall mean Avg Acc (Ā): {:.2f}% | mean Last Acc (A_B): {:.2f}%\n".format(
                        mean_avg, mean_last
                    )
                )
            by_f = {}
            for r in all_results:
                d = r.get("avg_last_fuse_by_fuse") or {}
                for fv, v in d.items():
                    try:
                        ff = float(fv)
                    except Exception:
                        continue
                    by_f.setdefault(ff, {"avg": [], "last": []})
                    by_f[ff]["avg"].append(float(v.get("avg", 0.0)))
                    by_f[ff]["last"].append(float(v.get("last", 0.0)))
            if by_f:
                f.write("\nOverall mean (by fuse) 融合:\n")
                for ff in sorted(by_f.keys()):
                    avgs = by_f[ff]["avg"]
                    lasts = by_f[ff]["last"]
                    if avgs and lasts:
                        f.write(
                            "  fuse=%.4f: mean Avg Acc (Ā) = %.2f%% | mean Last Acc (A_B) = %.2f%%\n"
                            % (float(ff), sum(avgs) / len(avgs), sum(lasts) / len(lasts))
                        )

    # CSV summary (DomainBed-style)
    import csv
    csv_path = os.path.join(log_dir, "results_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "val_domain", "seed", "train_domains", "classes", "tasks", "increment",
            "avg_acc", "last_acc", "task_top1_list",
            "fuse_list", "avg_last_fuse_by_fuse",
        ])
        for r in all_results:
            fuse_list = r.get("fuse_list") or []
            fuse_str = ",".join("%.4f" % float(x) for x in fuse_list) if fuse_list else ""
            by_fuse = r.get("avg_last_fuse_by_fuse") or {}
            by_fuse_parts = []
            for fv, d in sorted(by_fuse.items(), key=lambda kv: float(kv[0])):
                by_fuse_parts.append("fuse=%.4f:avg=%.2f,last=%.2f" % (float(fv), float(d.get("avg", 0.0)), float(d.get("last", 0.0))))
            w.writerow([
                r["val_domain"], r["seed"], "+".join(r["train_domains"]),
                r["classes"], r["tasks"], r["increment"],
                "{:.2f}".format(r["avg_acc"]), "{:.2f}".format(r["last_acc"]),
                "+".join("{:.2f}".format(x) for x in r["task_top1_list"]),
                fuse_str,
                " | ".join(by_fuse_parts),
            ])
        if all_results:
            mean_avg = sum(r["avg_acc"] for r in all_results) / len(all_results)
            mean_last = sum(r["last_acc"] for r in all_results) / len(all_results)
            w.writerow([
                "OVERALL_AVG", "ALL", "ALL", all_results[0]["classes"], all_results[0]["tasks"],
                all_results[0]["increment"], "{:.2f}".format(mean_avg), "{:.2f}".format(mean_last),
                "N/A", "", "",
            ])

    logging.info("=" * 60)
    logging.info("CIDG run finished. Logs and curves saved to: %s", log_dir)
    if all_results:
        mean_avg = sum(r["avg_acc"] for r in all_results) / len(all_results)
        mean_last = sum(r["last_acc"] for r in all_results) / len(all_results)
        fuse_results = [r for r in all_results if r.get("avg_acc_fuse") is not None]
        if fuse_results:
            mean_avg_fuse = sum(r["avg_acc_fuse"] for r in fuse_results) / len(fuse_results)
            mean_last_fuse = sum(r["last_acc_fuse"] for r in fuse_results) / len(fuse_results)
            logging.info(
                "Overall mean Avg Acc (Ā) CIL: %.2f%% | mean Last Acc (A_B) CIL: %.2f%%",
                mean_avg, mean_last,
            )
            logging.info(
                "Overall mean Avg Acc (Ā) 融合: %.2f%% | mean Last Acc (A_B) 融合: %.2f%%",
                mean_avg_fuse, mean_last_fuse,
            )
        else:
            logging.info(
                "Overall mean Avg Acc (Ā): %.2f%% | mean Last Acc (A_B): %.2f%%",
                mean_avg, mean_last,
            )
        by_f = {}
        for r in all_results:
            d = r.get("avg_last_fuse_by_fuse") or {}
            for fv, v in d.items():
                try:
                    ff = float(fv)
                except Exception:
                    continue
                by_f.setdefault(ff, {"avg": [], "last": []})
                by_f[ff]["avg"].append(float(v.get("avg", 0.0)))
                by_f[ff]["last"].append(float(v.get("last", 0.0)))
        if by_f:
            for ff in sorted(by_f.keys()):
                avgs = by_f[ff]["avg"]
                lasts = by_f[ff]["last"]
                if avgs and lasts:
                    logging.info(
                        "Overall mean (融合) fuse=%.4f | mean Avg Acc (Ā): %.2f%% | mean Last Acc (A_B): %.2f%%",
                        float(ff),
                        float(sum(avgs) / len(avgs)),
                        float(sum(lasts) / len(lasts)),
                    )
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
