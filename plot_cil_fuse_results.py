import os
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(results_root):
    """
    从 results/cil_fuse 目录加载所有 CIL 实验结果。

    目录结构约定：
    results/cil_fuse/{dataset}/{model}_seed{seed}_init{init}_inc{inc}.csv
    """
    data = defaultdict(lambda: defaultdict(list))  # {dataset: {model: [records...]}}

    for dataset in sorted(os.listdir(results_root)):
        ds_dir = os.path.join(results_root, dataset)
        if not os.path.isdir(ds_dir):
            continue
        for fname in sorted(os.listdir(ds_dir)):
            if not fname.endswith(".csv"):
                continue
            path = os.path.join(ds_dir, fname)
            # model_name 在文件名中 "_" 之前部分
            base = os.path.splitext(fname)[0]
            model_name = base.split("_seed")[0]
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                records = []
                for row in reader:
                    try:
                        task_idx = int(row.get("task_idx", 0))
                        num_classes = float(row.get("num_classes", 0))
                        cil_top1 = float(row.get("cil_top1", 0))
                    except ValueError:
                        continue
                    # 所有 fuse_* 列
                    fuse_cols = {k: float(v) for k, v in row.items() if k.startswith("fuse_") and v not in ("", None)}
                    records.append(
                        {
                            "task_idx": task_idx,
                            "num_classes": num_classes,
                            "cil_top1": cil_top1,
                            "fuse_cols": fuse_cols,
                        }
                    )
            if records:
                data[dataset][model_name].append(records)

    return data


def pick_best_fuse_curve(records):
    """
    对同一个实验（同一 CSV，records 为多行）选择一个 fuse 列作为“最佳融合曲线”：
    规则：选最终阶段 top1 最高的 fuse_* 列。
    """
    if not records:
        return None, None

    # 统计每个 fuse_* 列在最后一个 task 的值
    last_rec = records[-1]
    fuse_cols = last_rec["fuse_cols"]
    if not fuse_cols:
        return None, None

    # 选最后一阶段准确率最高的 fuse
    best_fuse_name, _ = max(fuse_cols.items(), key=lambda kv: kv[1])

    num_classes = [r["num_classes"] for r in records]
    cil_curve = [r["cil_top1"] for r in records]
    fuse_curve = [r["fuse_cols"].get(best_fuse_name, None) for r in records]

    # 从列名中解析出数值 fuse
    try:
        fuse_val = float(best_fuse_name.replace("fuse_", ""))
    except Exception:
        fuse_val = best_fuse_name

    return {
        "num_classes": num_classes,
        "cil_curve": cil_curve,
        "fuse_curve": fuse_curve,
        "fuse_name": best_fuse_name,
        "fuse_val": fuse_val,
    }, fuse_val


def plot_dataset(dataset, models_dict, save_dir):
    """
    对单个数据集绘图：
    - 颜色区分模型
    - 实线：融合前 CIL
    - 虚线：融合后（选择最佳 fuse）
    """
    if not models_dict:
        return

    plt.figure(figsize=(4.0, 3.0), dpi=150)
    colors = plt.cm.tab10.colors

    handles = []
    labels = []

    for idx, (model_name, runs) in enumerate(sorted(models_dict.items())):
        # 对同一模型、同一数据集，如有多次实验，这里简单取第一份
        records = runs[0]
        curve_info, fuse_val = pick_best_fuse_curve(records)
        if curve_info is None:
            continue

        num_classes = curve_info["num_classes"]
        cil_curve = curve_info["cil_curve"]
        fuse_curve = curve_info["fuse_curve"]

        c = colors[idx % len(colors)]

        # 实线：CIL
        h1, = plt.plot(
            num_classes,
            cil_curve,
            linestyle="-",
            marker="o",
            color=c,
            linewidth=1.8,
            markersize=3,
        )
        # 虚线：融合后
        h2, = plt.plot(
            num_classes,
            fuse_curve,
            linestyle="--",
            marker="o",
            color=c,
            linewidth=1.8,
            markersize=3,
        )
        handles.extend([h1, h2])
        labels.extend(
            [
                "{} (CIL)".format(model_name),
                "{} (fuse={})".format(model_name, fuse_val),
            ]
        )

    plt.xlabel("Number of Classes")
    plt.ylabel("Accuracy (%)")
    plt.title(dataset)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    # 图例放在下方，不挡曲线
    plt.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        fontsize=7,
        frameon=False,
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "{}.png".format(dataset))
    plt.savefig(save_path, dpi=150)
    plt.close()
    print("Saved plot for dataset {} to {}".format(dataset, save_path))


def main():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(this_dir, "results", "cil_fuse")
    if not os.path.isdir(results_root):
        print("results 目录不存在：{}".format(results_root))
        return

    all_data = load_results(results_root)
    plots_dir = os.path.join(results_root, "plots")
    for dataset, models_dict in sorted(all_data.items()):
        if not models_dict:
            continue
        plot_dataset(dataset, models_dict, plots_dir)


if __name__ == "__main__":
    main()

