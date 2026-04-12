import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(args["model_name"], args["dataset"], 
        init_cls, args["increment"], args["prefix"], args["seed"],args["backbone_type"],)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(args["dataset"],args["shuffle"],args["seed"],args["init_cls"],args["increment"], )
    model = factory.get_model(args["model_name"], args)
    model.save_dir = logs_name

    # CIL + ZS 融合 / 保存 zs 预测 (CIL main runner)
    save_zs_predictions = bool(args.get("save_zs_predictions", False))
    is_zs_clip = (args.get("model_name") or "").lower() == "zs_clip"
    _fuse = args.get("fuse", None)
    if isinstance(_fuse, (list, tuple)):
        _fuse_list = [float(x) for x in _fuse]
    elif _fuse is None:
        _fuse_list = []
    else:
        _fuse_list = [float(_fuse)]
    _fuse_list = sorted(set(_fuse_list))
    use_fuse = (len(_fuse_list) > 0) and (not is_zs_clip)
    _C3BOX_ROOT = os.path.dirname(os.path.abspath(__file__))
    zs_result_dir = os.path.join(
        _C3BOX_ROOT,
        "zs_result",
        args["dataset"],
        "seed{}_init{}_inc{}".format(args["seed"], args["init_cls"], args["increment"]),
    )
    if save_zs_predictions and not is_zs_clip:
        logging.warning("--save_zs_predictions is ignored when model_name is not zs_clip.")
        save_zs_predictions = False
    if save_zs_predictions:
        logging.info("Save zs predictions: ON -> writing to %s/task_*.csv", zs_result_dir)
    if use_fuse:
        logging.info(
            "Fuse ON: new_logits = CIL_logits + fuse * ZS_logits | fuse = %s | load dir: %s",
            ",".join("{:.4f}".format(x) for x in _fuse_list),
            zs_result_dir,
        )
        print("  [融合] 将从此目录加载 ZS logits: %s (需与 zs_clip --save_zs_predictions 时的 dataset/seed/init_cls/increment 一致)" % zs_result_dir, flush=True)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    zs_seen_curve, zs_unseen_curve, zs_harmonic_curve, zs_total_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}, {"top1": [], "top5": []}, {"top1": [], "top5": []}
    vit_curve_fuse = {f: {"top1": []} for f in _fuse_list}

    for task in range(data_manager.nb_tasks):
      #  logging.info("All params: {}".format(count_parameters(model._network)))
      #  logging.info(
      #      "Trainable params: {}".format(count_parameters(model._network, True))
      #  )
        model.incremental_train(data_manager)

        # Save zs_clip predictions for this incremental stage
        if save_zs_predictions and is_zs_clip and hasattr(model, "get_zs_predictions_and_logits"):
            try:
                y_pred_zs, y_true_zs, logits = model.get_zs_predictions_and_logits()
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

        # Load zs_result for fusion (other models)
        if use_fuse:
            model.zs_gate_table = None
            try:
                import pandas as pd
                csv_path = os.path.join(zs_result_dir, "task_{}.csv".format(task))
                if os.path.isfile(csv_path):
                    model.zs_gate_table = pd.read_csv(csv_path)
                else:
                    logging.warning(
                        "fuse 已设置但缺少 %s；本 task 不融合。请用相同 dataset=%s seed=%s init_cls=%s increment=%s 先跑: python main.py --config=./exps/zs_clip.json --save_zs_predictions",
                        csv_path, args["dataset"], args["seed"], args["init_cls"], args["increment"],
                    )
                    print("  [提示] 融合需要 zs_result CSV，当前查找: %s (未找到，请用相同 init_cls/increment 先跑 zs_clip --save_zs_predictions)" % csv_path, flush=True)
            except Exception as e:
                logging.warning("Failed to load zs_result for fusion: %s", e)

        result = model.eval_task()
        if isinstance(result, (list, tuple)):
            cnn_accy = result[0]
            nme_accy = result[1] if len(result) > 1 else None
            fuse_accy = result[2] if len(result) > 2 else None
        else:
            cnn_accy, nme_accy, fuse_accy = result, None, None
        model.after_task()

        cil_top1 = float(cnn_accy["top1"])
        logging.info("Task %d/%d | Test top1 (融合前 CIL): %.2f%%", task + 1, data_manager.nb_tasks, cil_top1)
        print("  [融合前] Task %d/%d 纯 CIL top1: %.2f%%" % (task + 1, data_manager.nb_tasks, cil_top1), flush=True)
        if fuse_accy is not None and isinstance(fuse_accy, dict):
            for f in sorted([float(k) for k in fuse_accy.keys()]):
                d = fuse_accy.get(float(f))
                if isinstance(d, dict) and "top1" in d:
                    fuse_top1 = float(d["top1"])
                    logging.info("Task %d/%d | Test top1 (融合后 fuse=%.4f): %.2f%%",
                                 task + 1, data_manager.nb_tasks, float(f), fuse_top1)
                    print("  [融合后] Task %d/%d fuse=%.4f: %.2f%% (融合前 %.2f%%)"
                          % (task + 1, data_manager.nb_tasks, float(f), fuse_top1, cil_top1), flush=True)
        else:
            if use_fuse:
                print("  [融合后] 本 task 未加载到 zs_result CSV，未计算融合分数。请用相同 dataset/seed/init_cls/increment 先跑 zs_clip --save_zs_predictions。", flush=True)

        # 累积融合曲线
        if fuse_accy is not None and isinstance(fuse_accy, dict):
            for f in _fuse_list:
                d = fuse_accy.get(float(f))
                if isinstance(d, dict) and "top1" in d:
                    vit_curve_fuse[f]["top1"].append(float(d["top1"]))

        logging.info("CNN: {}".format(cnn_accy["grouped"]))

        cnn_curve["top1"].append(cnn_accy["top1"])
        cnn_curve["top5"].append(cnn_accy["top5"])

        logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
        logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

        print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
        logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))

    # 实验结束：打印融合前 CIL 与各 fuse 融合后的最终 top1 和平均准确率
    nb = data_manager.nb_tasks
    last_cil = cnn_curve["top1"][-1] if cnn_curve["top1"] else 0.0
    avg_cil = sum(cnn_curve["top1"]) / len(cnn_curve["top1"]) if cnn_curve["top1"] else 0.0
    print("", flush=True)
    print("======== 实验结束 ========", flush=True)
    print("  [融合前 CIL] 最终 top1: %.2f%% | 增量阶段平均: %.2f%%" % (last_cil, avg_cil), flush=True)
    if use_fuse:
        for f in _fuse_list:
            lst = vit_curve_fuse.get(f, {}).get("top1", [])
            if lst:
                last_f = lst[-1]
                avg_f = sum(lst) / len(lst)
                print("  [融合后] fuse=%.4f | 最终 top1: %.2f%% | 增量阶段平均: %.2f%%" % (float(f), last_f, avg_f), flush=True)
                logging.info("Fuse=%.4f | Last top1: %.2f%% | Avg Acc: %.2f%%", float(f), last_f, avg_f)
    print("=========================", flush=True)

    # 将每个阶段融合前/后的 top1 保存到 results 目录，便于后续绘图
    try:
        import csv as _csv
        _C3BOX_ROOT = os.path.dirname(os.path.abspath(__file__))
        results_root = os.path.join(
            _C3BOX_ROOT,
            "results",
            "cil_fuse",
            str(args.get("dataset", "unknown_dataset")),
        )
        os.makedirs(results_root, exist_ok=True)
        results_path = os.path.join(
            results_root,
            "{}_seed{}_init{}_inc{}.csv".format(
                str(args.get("model_name", "model")),
                str(args.get("seed", "0")),
                str(args.get("init_cls", "0")),
                str(args.get("increment", "0")),
            ),
        )
        # 构造表头：task_idx, num_classes, cil_top1, fuse_xxx...
        header = ["task_idx", "num_classes", "cil_top1"]
        for f in _fuse_list:
            header.append("fuse_{:.4f}".format(float(f)))
        with open(results_path, "w", newline="", encoding="utf-8") as _f:
            writer = _csv.writer(_f)
            writer.writerow(header)
            for task in range(nb):
                # 当前阶段的有效类别数：使用 DataManager 的累积类数
                try:
                    num_classes = int(data_manager.get_accumulate_tasksize(task))
                except Exception:
                    # 退化为用 init_cls + (task+1)*increment 近似
                    init_c = int(args.get("init_cls", 0))
                    inc = int(args.get("increment", 0))
                    num_classes = init_c + (task + 1) * inc if inc > 0 else init_c
                cil_val = float(cnn_curve["top1"][task]) if task < len(cnn_curve["top1"]) else ""
                row = [task, num_classes, cil_val]
                for f in _fuse_list:
                    lst = vit_curve_fuse.get(f, {}).get("top1", [])
                    val = float(lst[task]) if task < len(lst) else ""
                    row.append(val)
                writer.writerow(row)
        logging.info("Per-task CIL/fuse accuracies saved to %s", results_path)
    except Exception as e:
        logging.warning("Failed to save per-task CIL/fuse results: %s", e)
    
def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
