# CIDG

Research code for **Class-Incremental Learning (CIL)** and **Class-Incremental Domain Generalization (CIDG)**.

This repository extends the open-source **[C3Box](https://github.com/LAMDA-CL/C3Box)** toolbox (a CLIP-based class-incremental learning framework). It keeps reproducibility and evaluation for many CLIP-based and classic CIL / ViT prompting methods, and adds:

1. **CIDG evaluation** (`dg_bench/`): DomainBed-style benchmarks with a **leave-one-domain-out** protocol—train on source domains and evaluate class-incremental learning **only on the held-out test domain**, with forgetting curves and logs.
2. **Zero-Shot Guided Fusion**: In the standard CIL loop, fuse **CLIP zero-shot logits** with **current CIL logits** using a scalar weight to improve cross-domain generalization. Save per-stage `zs_clip` predictions once, then load them for fusion with any CIL method.

---

## Dependencies

Aligned with upstream C3Box; adjust PyTorch builds for your CUDA version as needed.

- Python 3.8+
- [PyTorch](https://pytorch.org/) 2.x
- [torchvision](https://github.com/pytorch/vision)
- [timm](https://github.com/huggingface/pytorch-image-models)
- [open-clip](https://github.com/mlfoundations/open_clip)
- tqdm, numpy, scipy, easydict, pandas (fusion and result handling)

---

## Standard class-incremental experiments (CIL)

1. Pick or copy a JSON config under `exps/` and set `model_name`, `dataset`, `init_cls`, `increment`, `backbone_type`, etc.
2. Run:

```bash
python main.py --config=./exps/<config_name>.json
```

Supported `model_name` values match C3Box, e.g. `zs_clip`, `simplecil`, `foster`, `memo`, `l2p`, `dual`, `coda`, `ease`, `aper_*`, `tuna`, `rapf`, `clg_cbm`, `proof`, `engine`, `bofa` (see `utils/factory.py` and files under `exps/`).

---

## CIDG: domain-generalization CIL (`dg_bench`)

Organize datasets under a **root folder** following common DomainBed layouts (e.g. `PACS/`, `VLCS/`, …). Then run:

```bash
python dg_bench/runner.py --dataset_root <DATA_ROOT> --dataset_name PACS --model_name proof --device 0
```

- `--dataset_name`: `PACS`, `VLCS`, `OfficeHome`, `terra_incognita`, or `domain_net`.
- `--val_domain`: fix one domain as test; omit to sweep all domains (leave-one-out).
- `--seed`: pass multiple seeds for multi-seed protocols (e.g. DomainBed-style runs).
- Logs go to `./log/<timestamp>_<run_name>/` with arguments, accuracies, and forgetting-curve plots.

See `python dg_bench/runner.py --help` for ablations and options (e.g. image augmentation).

---

## Zero-Shot Guided Fusion

Fusion rule: **`new_logits = CIL_logits + fuse * ZS_logits`**.

**Workflow:**

1. Run **`zs_clip`** with the **same** `dataset`, `seed`, `init_cls`, and `increment`, and enable saving zero-shot outputs:

```bash
python main.py --config=./exps/zs_clip_<dataset>.json --save_zs_predictions
```

Outputs: `zs_result/<dataset>/seed<seed>_init<init>_inc<inc>/task_*.csv`.

2. Train your CIL method and pass one or more **`fuse`** weights via CLI or JSON (CSVs above are loaded automatically):

```bash
python main.py --config=./exps/<your_method>.json --fuse 0.5 1.0 1.5
```

Each `fuse` value is logged as its own curve. Use `plot_cil_fuse_results.py` to aggregate or plot results under `results/cil_fuse` (see the script for details).

**CIDG path:** `dg_bench/runner.py` also supports `--fuse` / `--save_zs_predictions` with the same idea; keep paths and task splits consistent.

---

## Datasets (standard CIL)

- **CIFAR-100**: downloaded automatically by the code.
- Other sets (CUB-200, ImageNet-R, ObjectNet, Aircraft, Food, …): prepare locally. For **non-CIFAR** runs, set your `train/` and `val/` roots in `download_data` inside `utils/data.py`.

Licensing and redistribution of each dataset follow the original publishers; this repo only provides loading and evaluation code.

---

## Citation

### C3Box (required when using their code or setup)

If you build on C3Box, please cite:

```bibtex
@article{sun2026c3box,
    title={C3Box: A CLIP-based Class-Incremental Learning Toolbox},
    author={Sun, Hao and Zhou, Da-Wei},
    journal={arXiv preprint arXiv:2601.20852},
    year={2026}
}
```

The C3Box README also lists related surveys (e.g. Zhou et al., IJCAI 2024; Zhou et al., TPAMI 2024)—cite them if relevant.

### This repository’s CIDG extensions

If you publish work that uses the **CIDG benchmark or fusion modules** added here, state that the code extends C3Box and keep the C3Box citation above; add your own paper’s bib entry for the new contribution.

---

## Acknowledgments

- Core CIL implementations and baselines from **[C3Box](https://github.com/LAMDA-CL/C3Box)** (LAMDA-CL).
- Engineering ideas from community projects such as **[PyCIL](https://github.com/G-U-N/PyCIL)** and **[LAMDA-PILOT](https://github.com/LAMDA-CL/LAMDA-PILOT)**.

---

## License

See `LICENSE` (MIT). Third-party datasets and pretrained weights remain under their respective licenses.

---

**Repository:** https://github.com/xhgSh/CIDG
