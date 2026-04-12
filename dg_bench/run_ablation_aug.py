"""
Ablation: 风格 / 形状 / 形变 三种增强的组合研究，train-time 与 test-time 分开消融。

运行参数（可组合多组消融）：
  --dataset_name  数据集：PACS / VLCS / OfficeHome / terra_incognita / domain_net（默认 terra_incognita）
  --model_name    模型：C3Box 模型名（默认 aper_vpt）
  --increment     每 task 新增类别数（默认 2）

消融组合（8 组）：baseline, style, shape, deformation, style_shape, style_deformation, shape_deformation, all
默认对全部域做 leave-one-out CIDG，实验流程对齐；--val_domain 指定单域可快速测试。

Usage:
  python -m dg_bench.run_ablation_aug --dataset_root /path/to/datasets
  python -m dg_bench.run_ablation_aug --dataset_root /path/to/datasets --dataset_name OfficeHome --model_name aper_vpt --increment 5
  python -m dg_bench.run_ablation_aug --dataset_root /path/to/datasets --val_domain Art
"""
import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple

# C3Box root
_C3BOX_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _C3BOX_ROOT not in sys.path:
    sys.path.insert(0, _C3BOX_ROOT)


# 8 种消融组合名称（风格/形状/形变）
ABLATION_CONFIG_NAMES = [
    "baseline", "style", "shape", "deformation",
    "style_shape", "style_deformation", "shape_deformation", "all",
]


def _run_ablation_plan(
    args,
    dataset_name: str,
    model_name: str,
    domains: List[str],
    eval_plan: List[str],
    configs: List[Dict[str, Any]],
    base_argv: List[str],
    log_dir: str,
    ablation_tag: str,
) -> List[Dict[str, Any]]:
    from dg_bench.runner import build_args_dict, run_once, parse_args as runner_parse_args
    results: List[Dict[str, Any]] = []
    for config in configs:
        # 每进入一个新的消融组合时醒目提示
        sep = "=" * 60
        prompt = ">>> 当前消融组合: [%s] %s" % (ablation_tag, config["name"])
        print("", flush=True)
        print(sep, flush=True)
        print("  " + prompt, flush=True)
        print(sep, flush=True)
        logging.info("")
        logging.info(sep)
        logging.info("  [%s] 消融组合: %s", ablation_tag, config["name"])
        logging.info(sep)

        argv = list(base_argv)
        if config.get("aug_enable"):
            argv += [
                "--aug_enable",
                "--aug_methods", str(config.get("aug_methods", "all")),
                "--aug_tta_methods", str(config.get("aug_tta_methods", "all")),
                "--aug_train_views", str(config.get("aug_train_views", 0)),
                "--aug_test_views", str(config.get("aug_test_views", 0)),
                "--aug_magnitude", str(config.get("aug_magnitude", 3.0)),
            ]
        sys.argv = ["runner.py"] + argv
        cli = runner_parse_args()
        args_dict = build_args_dict(cli)
        args_dict["save_dir"] = os.path.join(log_dir, "run_{}_{}".format(ablation_tag, config["name"]))

        # 与所有其他组合完全一致的实验流程：同一 eval_plan 顺序 × 同一 seeds 顺序，便于对齐对比
        for vd in eval_plan:
            train_domains = [d for d in domains if d != vd]
            for seed in args.seeds:
                logging.info("  [%s] config: %s | Test domain: %s | Seed: %s",
                             ablation_tag, config["name"], vd, seed)
                res = run_once(args_dict, train_domains, vd, seed)
                res["aug_config"] = config["name"]
                res["ablation_type"] = ablation_tag
                results.append(res)
    return results


# 与 runner 一致的数据集/模型选项，供消融命令行使用
ABLATION_DATASET_CHOICES = ["PACS", "VLCS", "OfficeHome", "terra_incognita", "domain_net"]


def main():
    parser = argparse.ArgumentParser(
        description="Ablation: train-time vs test-time aug. 数据集、模型、increment 为运行参数，可组合多组消融。"
    )
    # 三个主要运行参数：数据集、模型、增量数
    parser.add_argument("--dataset_name", type=str, default="terra_incognita",
                        choices=ABLATION_DATASET_CHOICES,
                        help="数据集名称（默认 terra_incognita）")
    parser.add_argument("--model_name", type=str, default="aper_vpt",
                        help="C3Box 模型名（默认 aper_vpt）")
    parser.add_argument("--increment", type=int, default=2,
                        help="每 task 新增类别数（默认 2）")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="数据根目录，其下需有对应数据集文件夹（如 office_home、terra_incognita）")
    parser.add_argument("--val_domain", type=str, default=None,
                        help="指定单域时只在该域做 leave-one-out，用于快速测试；未指定时默认跑完全部域（实验对齐）")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--aug_train_views", type=int, default=2,
                        help="Extra train views when train ablation uses aug")
    parser.add_argument("--aug_test_views", type=int, default=4,
                        help="TTA views when TTA ablation uses aug")
    parser.add_argument("--aug_magnitude", type=float, default=3.0)
    parser.add_argument("--ablation", type=str, default="both",
                        choices=["train_only", "tta_only", "both"],
                        help="train_only: 只消融训练增强; tta_only: 只消融 TTA; both: 两者都跑")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="日志目录；未指定时自动生成 log/ablation_aug_{dataset}_{model}_inc{increment}_{timestamp}")
    parser.add_argument("--init_cls", type=int, default=0,
                        help="Initial number of classes (与 runner 一致)")
    args = parser.parse_args()

    from dg_bench.runner import discover_domains, _C3BOX_ROOT, C3BOX_MODEL_NAMES
    if args.model_name not in C3BOX_MODEL_NAMES:
        raise ValueError("model_name must be one of: %s" % ", ".join(C3BOX_MODEL_NAMES))

    dataset_name = args.dataset_name
    model_name = args.model_name
    domains = discover_domains(args.dataset_root, dataset_name)
    # 默认跑完全部域，保证各消融组合实验流程一致、可对比；--val_domain 指定时只跑单域
    if args.val_domain:
        eval_plan = [args.val_domain]
        if args.val_domain not in domains:
            raise ValueError("val_domain {} not in {}".format(args.val_domain, domains))
    else:
        eval_plan = list(domains)
    logging.info("Ablation 数据集=%s 模型=%s increment=%d", dataset_name, model_name, args.increment)
    logging.info("Ablation 测试域 (eval_plan，所有组合统一): %s (共 %d 个域)", eval_plan, len(eval_plan))
    logging.info("Seeds (所有组合统一): %s", args.seeds)

    log_dir = args.log_dir or os.path.join(
        _C3BOX_ROOT, "log",
        "ablation_aug_{}_{}_inc{}_{}".format(
            dataset_name, model_name, args.increment, datetime.now().strftime("%Y%m%d_%H%M%S")))
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, "ablation.log"), encoding="utf-8"),
        ],
    )

    # 与正常实验一致：init_cls/increment 默认 0/2，与 runner.py 常用参数一致，baseline 才可比
    base_argv = [
        "--dataset_root", args.dataset_root,
        "--dataset_name", dataset_name,
        "--model_name", model_name,
        "--vpt_type", "shallow",
        "--init_cls", str(args.init_cls),
        "--increment", str(args.increment),
        "--seed", *[str(s) for s in args.seeds],
    ]
    if args.val_domain:
        base_argv += ["--val_domain", args.val_domain]

    # 风格 / 形状 / 形变 三种组合，共 8 组
    def _make_config(name: str, aug_enable: bool, train_views: int, test_views: int, methods: str) -> Dict[str, Any]:
        c = {"name": name, "aug_enable": aug_enable, "aug_magnitude": args.aug_magnitude}
        if aug_enable:
            c["aug_methods"] = c["aug_tta_methods"] = methods
            c["aug_train_views"] = train_views
            c["aug_test_views"] = test_views
        return c

    # Train-time ablation: 只变训练增强，TTA 关闭 (test_views=0)
    train_configs: List[Dict[str, Any]] = [
        _make_config("baseline", False, 0, 0, "all"),
        _make_config("style", True, args.aug_train_views, 0, "style"),
        _make_config("shape", True, args.aug_train_views, 0, "shape"),
        _make_config("deformation", True, args.aug_train_views, 0, "deformation"),
        _make_config("style_shape", True, args.aug_train_views, 0, "style_shape"),
        _make_config("style_deformation", True, args.aug_train_views, 0, "style_deformation"),
        _make_config("shape_deformation", True, args.aug_train_views, 0, "shape_deformation"),
        _make_config("all", True, args.aug_train_views, 0, "all"),
    ]

    # TTA ablation: 只变测试增强，训练扩张关闭 (train_views=0)
    tta_configs: List[Dict[str, Any]] = [
        _make_config("baseline", False, 0, 0, "all"),
        _make_config("style", True, 0, args.aug_test_views, "style"),
        _make_config("shape", True, 0, args.aug_test_views, "shape"),
        _make_config("deformation", True, 0, args.aug_test_views, "deformation"),
        _make_config("style_shape", True, 0, args.aug_test_views, "style_shape"),
        _make_config("style_deformation", True, 0, args.aug_test_views, "style_deformation"),
        _make_config("shape_deformation", True, 0, args.aug_test_views, "shape_deformation"),
        _make_config("all", True, 0, args.aug_test_views, "all"),
    ]

    all_results: List[Dict[str, Any]] = []
    if args.ablation in ("train_only", "both"):
        all_results.extend(_run_ablation_plan(
            args, dataset_name, model_name, domains, eval_plan,
            train_configs, base_argv, log_dir, "train",
        ))
    if args.ablation in ("tta_only", "both"):
        all_results.extend(_run_ablation_plan(
            args, dataset_name, model_name, domains, eval_plan,
            tta_configs, base_argv, log_dir, "tta",
        ))

    # Summary per (ablation_type, aug_config): 平均准确率 Avg Acc (Ā) 与 最终准确率 Last Acc (A_B)
    by_key_last: Dict[Tuple[str, str], List[float]] = {}
    by_key_avg: Dict[Tuple[str, str], List[float]] = {}
    for r in all_results:
        key = (r["ablation_type"], r["aug_config"])
        if key not in by_key_last:
            by_key_last[key] = []
            by_key_avg[key] = []
        by_key_last[key].append(r.get("last_acc", r.get("final_task_top1", 0)))
        by_key_avg[key].append(r.get("avg_acc", (sum(r["task_top1_list"]) / len(r["task_top1_list"])) if r.get("task_top1_list") else 0))

    summary = {
        "dataset": dataset_name,
        "model": model_name,
        "ablation_mode": args.ablation,
        "eval_plan": eval_plan,
        "seeds": args.seeds,
        "ablation_summary": {},
        "all_results": [
            {"ablation_type": r["ablation_type"], "aug_config": r["aug_config"],
             "val_domain": r["val_domain"], "seed": r["seed"],
             "avg_acc": r.get("avg_acc"), "last_acc": r.get("last_acc", r.get("final_task_top1")),
             "final_task_top1": r.get("final_task_top1"), "task_top1_list": r["task_top1_list"]}
            for r in all_results
        ],
    }
    for (ablation_type, config_name) in by_key_last:
        key = "{}__{}".format(ablation_type, config_name)
        values_last = by_key_last[(ablation_type, config_name)]
        values_avg = by_key_avg[(ablation_type, config_name)]
        mean_last = sum(values_last) / len(values_last) if values_last else 0
        std_last = (sum((x - mean_last) ** 2 for x in values_last) / len(values_last)) ** 0.5 if len(values_last) > 1 else 0
        mean_avg = sum(values_avg) / len(values_avg) if values_avg else 0
        std_avg = (sum((x - mean_avg) ** 2 for x in values_avg) / len(values_avg)) ** 0.5 if len(values_avg) > 1 else 0
        summary["ablation_summary"][key] = {
            "ablation_type": ablation_type,
            "config": config_name,
            "mean_avg_acc": mean_avg,
            "std_avg_acc": std_avg,
            "mean_last_acc": mean_last,
            "std_last_acc": std_last,
            "mean_final_top1": mean_last,
            "std_final_top1": std_last,
            "runs": len(values_last),
            "values_avg_acc": values_avg,
            "values_last_acc": values_last,
        }

    summary_path = os.path.join(log_dir, "ablation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(log_dir, "ablation_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("ablation_type,aug_config,val_domain,seed,avg_acc,last_acc,task_top1_list\n")
        for r in all_results:
            task_list_str = "+".join("{:.2f}".format(x) for x in r["task_top1_list"])
            avg = r.get("avg_acc")
            last = r.get("last_acc", r.get("final_task_top1"))
            if avg is None and r.get("task_top1_list"):
                avg = sum(r["task_top1_list"]) / len(r["task_top1_list"])
            f.write("{},{},{},{},{:.2f},{:.2f},{}\n".format(
                r["ablation_type"], r["aug_config"], r["val_domain"], r["seed"],
                avg or 0, last or 0, task_list_str))
        f.write("\n# Per (ablation_type, config) mean: Avg Acc (Ā), Last Acc (A_B)\n")
        for key, data in summary["ablation_summary"].items():
            f.write("# {}: Avg Acc mean={:.2f}% std={:.2f}% | Last Acc mean={:.2f}% std={:.2f}% n={}\n".format(
                key, data["mean_avg_acc"], data["std_avg_acc"], data["mean_last_acc"], data["std_last_acc"], data["runs"]))

    # Generate comparison plots
    def _plot_ablation_comparison(
        results: List[Dict[str, Any]],
        ablation_type: str,
        configs: List[str],
        save_path: str,
        title: str,
    ):
        """Plot forgetting curves for all configs in one figure."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.ticker import MaxNLocator
        except ImportError:
            logging.warning("matplotlib not available; skipping ablation comparison plot.")
            return
        # Group by config, average task_top1_list over (domain, seed)
        by_config: Dict[str, List[List[float]]] = {}
        for r in results:
            if r["ablation_type"] != ablation_type:
                continue
            cfg = r["aug_config"]
            if cfg not in by_config:
                by_config[cfg] = []
            by_config[cfg].append(r["task_top1_list"])
        if not by_config:
            return
        n_tasks = len(by_config[list(by_config.keys())[0]][0])
        plt.figure(figsize=(max(10, len(configs) * 1.1), 5))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
        for i, cfg in enumerate(configs):
            if cfg not in by_config:
                continue
            lists = by_config[cfg]
            avg_list = [sum(lists[s][t] for s in range(len(lists))) / len(lists) for t in range(n_tasks)]
            tasks = list(range(1, n_tasks + 1))
            plt.plot(tasks, avg_list, marker="o", linewidth=2, markersize=6,
                    color=colors[i % len(colors)], label=cfg)
        plt.xlabel("Task (after learning up to this task)", fontsize=11)
        plt.ylabel("Accuracy on test domain (%)", fontsize=11)
        plt.title(title, fontsize=12)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        logging.info("Saved ablation comparison plot: %s", save_path)

    # Plot train-time ablation comparison (风格/形状/形变 8 组)
    if args.ablation in ("train_only", "both"):
        train_results = [r for r in all_results if r["ablation_type"] == "train"]
        if train_results:
            _plot_ablation_comparison(
                train_results,
                "train",
                ABLATION_CONFIG_NAMES,
                os.path.join(log_dir, "ablation_comparison_train.png"),
                "Train-time Ablation: style/shape/deformation ({} on {})".format(model_name, dataset_name),
            )
    # Plot TTA ablation comparison
    if args.ablation in ("tta_only", "both"):
        tta_results = [r for r in all_results if r["ablation_type"] == "tta"]
        if tta_results:
            _plot_ablation_comparison(
                tta_results,
                "tta",
                ABLATION_CONFIG_NAMES,
                os.path.join(log_dir, "ablation_comparison_tta.png"),
                "TTA Ablation: style/shape/deformation ({} on {})".format(model_name, dataset_name),
            )

    # 3) Bar chart: mean final top1 per (ablation_type, config)
    def _plot_ablation_bar_summary():
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            return
        if not summary.get("ablation_summary"):
            return
        keys = list(summary["ablation_summary"].keys())
        keys.sort(key=lambda k: (summary["ablation_summary"][k]["ablation_type"], summary["ablation_summary"][k]["config"]))
        labels = [summary["ablation_summary"][k]["config"] + "\n(" + summary["ablation_summary"][k]["ablation_type"] + ")" for k in keys]
        means_avg = [summary["ablation_summary"][k]["mean_avg_acc"] for k in keys]
        stds_avg = [summary["ablation_summary"][k]["std_avg_acc"] for k in keys]
        means_last = [summary["ablation_summary"][k]["mean_last_acc"] for k in keys]
        stds_last = [summary["ablation_summary"][k]["std_last_acc"] for k in keys]
        n = len(labels)
        x = np.arange(n)
        width = 0.35
        plt.figure(figsize=(max(14, n * 1.5), 5))
        plt.bar(x - width / 2, means_avg, width, yerr=stds_avg, capsize=3, label="Avg Acc ($\\bar{A}$)", color="#1f77b4")
        plt.bar(x + width / 2, means_last, width, yerr=stds_last, capsize=3, label="Last Acc ($A_B$)", color="#ff7f0e")
        plt.xticks(x, labels, fontsize=9, rotation=15)
        plt.ylabel("Accuracy (%)", fontsize=11)
        plt.title("Ablation Summary: {} on {} (init_cls={}, increment={})".format(
            model_name, dataset_name, args.init_cls, args.increment))
        plt.legend(loc="upper right", fontsize=10)
        plt.tight_layout()
        bar_path = os.path.join(log_dir, "ablation_bar_summary.png")
        plt.savefig(bar_path, dpi=150)
        plt.close()
        logging.info("Saved ablation bar summary: %s", bar_path)

    _plot_ablation_bar_summary()

    # 打印所有组合的实验结果表（对齐实验流程后的汇总）
    def _print_results_table():
        if not summary.get("ablation_summary"):
            return
        keys = list(summary["ablation_summary"].keys())
        keys.sort(key=lambda k: (summary["ablation_summary"][k]["ablation_type"], summary["ablation_summary"][k]["config"]))
        # 表头：消融类型 | 配置 | Avg Acc (Ā) mean±std | Last Acc (A_B) mean±std | n
        col_ablation = "消融类型"
        col_config = "配置"
        col_avg = "Avg Acc (Ā)"
        col_last = "Last Acc (A_B)"
        col_n = "n"
        w_ablation = max(len(col_ablation), 8)
        w_config = max(len(col_config), max(len(summary["ablation_summary"][k]["config"]) for k in keys), 12)
        w_avg = 20
        w_last = 20
        w_n = 6
        sep_line = "+" + "-" * (w_ablation + 2) + "+" + "-" * (w_config + 2) + "+" + "-" * (w_avg + 2) + "+" + "-" * (w_last + 2) + "+" + "-" * (w_n + 2) + "+"
        header = "| {} | {} | {} | {} | {} |".format(
            col_ablation.ljust(w_ablation), col_config.ljust(w_config),
            col_avg.ljust(w_avg), col_last.ljust(w_last), col_n.ljust(w_n))
        lines = [
            "",
            "======== 所有消融组合实验结果表 (eval_plan={} 域, seeds={}) ========".format(len(eval_plan), args.seeds),
            sep_line,
            header,
            sep_line,
        ]
        for k in keys:
            d = summary["ablation_summary"][k]
            typ = d["ablation_type"]
            cfg = d["config"]
            avg_s = "{:.2f}±{:.2f}".format(d["mean_avg_acc"], d["std_avg_acc"])
            last_s = "{:.2f}±{:.2f}".format(d["mean_last_acc"], d["std_last_acc"])
            n_s = str(d["runs"])
            lines.append("| {} | {} | {} | {} | {} |".format(
                typ.ljust(w_ablation), cfg.ljust(w_config),
                avg_s.ljust(w_avg), last_s.ljust(w_last), n_s.ljust(w_n)))
        lines.append(sep_line)
        lines.append("")
        text = "\n".join(lines)
        print(text, flush=True)
        logging.info(text)
        table_path = os.path.join(log_dir, "ablation_results_table.txt")
        with open(table_path, "w", encoding="utf-8") as f:
            f.write(text)
        logging.info("Results table saved to %s", table_path)

    _print_results_table()

    logging.info("===== Ablation summary (风格/形状/形变 8 组) | Avg Acc (Ā) & Last Acc (A_B) =====")
    for typ in ("train", "tta"):
        if not any(r["ablation_type"] == typ for r in all_results):
            continue
        logging.info("--- %s-time ---", typ)
        for config_name in ABLATION_CONFIG_NAMES:
            k = "{}__{}".format(typ, config_name)
            if k in summary["ablation_summary"]:
                d = summary["ablation_summary"][k]
                logging.info("  %s: Avg Acc mean=%.2f%% std=%.2f%% | Last Acc mean=%.2f%% std=%.2f%% n=%d",
                             config_name, d["mean_avg_acc"], d["std_avg_acc"], d["mean_last_acc"], d["std_last_acc"], d["runs"])
    logging.info("Results saved to %s", log_dir)


if __name__ == "__main__":
    main()
