# CIDG

面向 **类增量学习（Class-Incremental Learning, CIL）** 与 **类增量域泛化（Class-Incremental Domain Generalization, CIDG）** 的研究代码库。

本仓库在开源工具箱 **[C3Box](https://github.com/LAMDA-CL/C3Box)**（基于 CLIP 的类增量学习框架）之上扩展实现，保留其对多种 CLIP 类增量方法与经典 CIL / ViT 提示类方法的复现与评测能力，并新增：

1. **CIDG 评测框架**（`dg_bench/`）：在 DomainBed 风格数据集上采用 **留一域（leave-one-domain-out）** 协议，在 **仅未见测试域** 上评估类增量过程，并记录遗忘曲线等日志。
2. **Zero-Shot Guided Fusion（零样本引导融合）**：在标准 CIL 主流程中，将 **CLIP 零样本 logits** 与 **当前 CIL 模型 logits** 按系数融合，用于提升跨域泛化表现；支持先保存 `zs_clip` 各阶段预测再在任意 CIL 方法上加载融合。

---

## 环境依赖

与上游 C3Box 一致，建议版本如下（可按本地 CUDA 调整 PyTorch 版本）：

- Python 3.8+
- [PyTorch](https://pytorch.org/) 2.x
- [torchvision](https://github.com/pytorch/vision)
- [timm](https://github.com/huggingface/pytorch-image-models)
- [open-clip](https://github.com/mlfoundations/open_clip)
- tqdm、numpy、scipy、easydict、pandas（用于融合与结果处理）

---

## 标准类增量实验（CIL）

1. 在 `exps/` 下选择或复制 JSON 配置，设置 `model_name`、`dataset`、`init_cls`、`increment`、`backbone_type` 等。
2. 运行：

```bash
python main.py --config=./exps/<配置名>.json
```

支持的 `model_name` 与 C3Box 一致，例如：`zs_clip`、`simplecil`、`foster`、`memo`、`l2p`、`dual`、`coda`、`ease`、`aper_*`、`tuna`、`rapf`、`clg_cbm`、`proof`、`engine`、`bofa` 等（以 `utils/factory.py` 与 `exps/` 中实际文件为准）。

---

## CIDG：域泛化类增量评测（`dg_bench`）

在 **数据集根目录** 下按 DomainBed 惯例组织各数据集子文件夹（如 `PACS/`、`VLCS/` 等），运行：

```bash
python dg_bench/runner.py --dataset_root <数据根目录> --dataset_name PACS --model_name proof --device 0
```

- `--dataset_name` 可选：`PACS`、`VLCS`、`OfficeHome`、`terra_incognita`、`domain_net`。
- `--val_domain`：指定某一域为测试域；省略则对所有域做留一域遍历。
- `--seed`：可传多个种子以符合 DomainBed 多随机种子协议。
- 日志默认写入 `./log/<时间戳>_<实验名>/`，含参数、准确率与遗忘曲线图等。

更多消融与增强（如图像增强相关参数）见 `dg_bench/runner.py` 的命令行说明。

---

## Zero-Shot Guided Fusion（零样本引导融合）

融合公式：**`new_logits = CIL_logits + fuse * ZS_logits`**。

**步骤概览：**

1. **先**用 `model_name=zs_clip` 在 **相同** `dataset`、`seed`、`init_cls`、`increment` 下运行，并开启保存零样本预测：

```bash
python main.py --config=./exps/zs_clip_<数据集>.json --save_zs_predictions
```

输出目录：`zs_result/<dataset>/seed<seed>_init<init>_inc<inc>/task_*.csv`。

2. **再**训练你的 CIL 方法，并通过 CLI 或 JSON 传入一个或多个 `fuse` 系数（将自动加载上述目录中的 CSV）：

```bash
python main.py --config=./exps/<你的方法>.json --fuse 0.5 1.0 1.5
```

多组 `fuse` 会分别记录曲线。可用 `plot_cil_fuse_results.py` 对 `results/cil_fuse` 等目录下的结果做汇总与可视化（见脚本内说明）。

**CIDG 路径**：`dg_bench/runner.py` 同样支持 `--fuse` / `--save_zs_predictions`，流程与上类似，需保证路径与任务划分一致。

---

## 数据集（标准 CIL）

- **CIFAR-100**：可由代码自动下载。
- 其余数据集（CUB-200、ImageNet-R、ObjectNet、Aircraft、Food 等）需自行准备；**非 CIFAR** 时请在 `utils/data.py` 的 `download_data` 中配置你的 `train/` 与 `val/` 路径。

各数据集版权与分发条款以原始发布方为准；本仓库仅提供加载与评测代码。

---

## 引用

### 使用本仓库时，请引用 C3Box 原论文（上游要求）

若你的工作基于或使用了 C3Box 代码与设定，请引用：

```bibtex
@article{sun2026c3box,
    title={C3Box: A CLIP-based Class-Incremental Learning Toolbox},
    author={Sun, Hao and Zhou, Da-Wei},
    journal={arXiv preprint arXiv:2601.20852},
    year={2026}
}
```

C3Box 文中亦建议关注类增量与持续学习综述，可按需引用其 README 中的 [Zhou et al., IJCAI 2024]、[Zhou et al., TPAMI 2024] 等条目。

### 若你发表基于 **本 CIDG 扩展** 的工作

请在文中说明代码基于 C3Box 扩展，并同时引用 C3Box；对本仓库新增的 CIDG 评测与融合模块，请按你的论文信息撰写 bibtex（此处不代写正式条目）。

---

## 致谢

- 核心 CIL 实现与基准来自 **[C3Box](https://github.com/LAMDA-CL/C3Box)**（LAMDA-CL）。
- 工程上参考 **[PyCIL](https://github.com/G-U-N/PyCIL)**、**[LAMDA-PILOT](https://github.com/LAMDA-CL/LAMDA-PILOT)** 等社区项目。

---

## 许可证

见仓库根目录 `LICENSE`（MIT）。使用第三方数据集与预训练权重时，请遵守其各自许可协议。

---

## 仓库地址

**https://github.com/xhgSh/CIDG**
