"""
Image-level data augmentation for CIL/DG: AugMix (13 ops) + hflip + RevisitingCIL (elastic, grid).
Train and test use the same full chain for consistency.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageEnhance

# ----- AugMix-style PIL ops (configurable image size) -----
def _int_param(level: float, maxval: float) -> int:
    return int(level * maxval / 10)

def _float_param(level: float, maxval: float) -> float:
    return float(level) * maxval / 10

def _sample_level(n: float) -> float:
    return np.random.uniform(low=0.1, high=n)

def _pil_op_autocontrast(pil_img: Image.Image, _level: float, _size: int) -> Image.Image:
    return ImageOps.autocontrast(pil_img)

def _pil_op_equalize(pil_img: Image.Image, _level: float, _size: int) -> Image.Image:
    return ImageOps.equalize(pil_img)

def _pil_op_posterize(pil_img: Image.Image, level: float, _size: int) -> Image.Image:
    level = _int_param(_sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)

def _pil_op_rotate(pil_img: Image.Image, level: float, size: int) -> Image.Image:
    degrees = _int_param(_sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)

def _pil_op_solarize(pil_img: Image.Image, level: float, _size: int) -> Image.Image:
    level = _int_param(_sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)

def _pil_op_shear_x(pil_img: Image.Image, level: float, size: int) -> Image.Image:
    level = _float_param(_sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((size, size), Image.AFFINE, (1, level, 0, 0, 1, 0), resample=Image.BILINEAR)

def _pil_op_shear_y(pil_img: Image.Image, level: float, size: int) -> Image.Image:
    level = _float_param(_sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((size, size), Image.AFFINE, (1, 0, 0, level, 1, 0), resample=Image.BILINEAR)

def _pil_op_translate_x(pil_img: Image.Image, level: float, size: int) -> Image.Image:
    level = _int_param(_sample_level(level), max(1, size // 3))
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((size, size), Image.AFFINE, (1, 0, level, 0, 1, 0), resample=Image.BILINEAR)

def _pil_op_translate_y(pil_img: Image.Image, level: float, size: int) -> Image.Image:
    level = _int_param(_sample_level(level), max(1, size // 3))
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((size, size), Image.AFFINE, (1, 0, 0, 0, 1, level), resample=Image.BILINEAR)

def _pil_op_color(pil_img: Image.Image, level: float, _size: int) -> Image.Image:
    level = _float_param(_sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)

def _pil_op_contrast(pil_img: Image.Image, level: float, _size: int) -> Image.Image:
    level = _float_param(_sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)

def _pil_op_brightness(pil_img: Image.Image, level: float, _size: int) -> Image.Image:
    level = _float_param(_sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)

def _pil_op_sharpness(pil_img: Image.Image, level: float, _size: int) -> Image.Image:
    level = _float_param(_sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)

def _pil_op_hflip(pil_img: Image.Image, _level: float, _size: int) -> Image.Image:
    return pil_img.transpose(Image.FLIP_LEFT_RIGHT)

# Presets: style = color/light; shape = geometry. Full = AugMix 13 + hflip (same as augmix-master augmentations_all + hflip)
PIL_OPS_STYLE = [
    _pil_op_autocontrast, _pil_op_equalize, _pil_op_posterize, _pil_op_solarize,
    _pil_op_color, _pil_op_contrast, _pil_op_brightness, _pil_op_sharpness,
]
PIL_OPS_SHAPE = [
    _pil_op_rotate, _pil_op_shear_x, _pil_op_shear_y,
    _pil_op_translate_x, _pil_op_translate_y, _pil_op_hflip,
]
# Full chain: all AugMix + hflip (14 PIL ops). RevisitingCIL elastic/grid applied on tensor after.
PIL_OPS_ALL = PIL_OPS_STYLE + PIL_OPS_SHAPE


# ----- Tensor ops (elastic, grid, hflip) for TTA -----
def _gaussian_kernel2d(kernel_size: int = 21, sigma: float = 3.0, device: Optional[torch.device] = None):
    ax = torch.arange(kernel_size, device=device, dtype=torch.float32) - (kernel_size - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)

def _smooth_displacement(disp: torch.Tensor, sigma: float = 8.0) -> torch.Tensor:
    kernel_size = int(2 * round(3 * sigma) + 1)
    kernel = _gaussian_kernel2d(kernel_size, sigma, device=disp.device)
    disp_x = F.conv2d(disp[:, 0:1], kernel, padding=kernel_size // 2)
    disp_y = F.conv2d(disp[:, 1:2], kernel, padding=kernel_size // 2)
    return torch.cat([disp_x, disp_y], dim=1)

def _make_base_grid(batch: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    ys = torch.linspace(-1, 1, height, device=device)
    xs = torch.linspace(-1, 1, width, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([grid_x, grid_y], dim=-1)
    return base.unsqueeze(0).repeat(batch, 1, 1, 1)

def _apply_flow(x: torch.Tensor, flow_xy: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = x.shape
    base = _make_base_grid(batch, height, width, x.device)
    grid = base + flow_xy.permute(0, 2, 3, 1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)

def elastic_deform_tensor(x: torch.Tensor, alpha_std: float = 0.08, sigma: float = 8.0) -> torch.Tensor:
    batch, _, height, width = x.shape
    alpha = torch.randn(batch, device=x.device) * alpha_std
    alpha = alpha.abs().view(batch, 1, 1, 1)
    disp = torch.randn(batch, 2, height, width, device=x.device)
    disp = _smooth_displacement(disp, sigma=sigma)
    disp = disp / (disp.abs().amax(dim=(2, 3), keepdim=True) + 1e-8) * alpha
    return _apply_flow(x, disp)

def grid_distortion_tensor(
    x: torch.Tensor, grid_rows: int = 4, grid_cols: int = 4, distort_std: float = 0.01
) -> torch.Tensor:
    batch, _, height, width = x.shape
    gh, gw = grid_rows + 1, grid_cols + 1
    distort = torch.abs(torch.randn(1, device=x.device) * distort_std)
    ctrl = (torch.rand(batch, 2, gh, gw, device=x.device) * 2 - 1) * distort
    dense = F.interpolate(ctrl, size=(height, width), mode="bicubic", align_corners=True)
    return _apply_flow(x, dense)

def hflip_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-1])


class ElasticGridTransform:
    """Apply elastic + grid (RevisitingCIL) to a single tensor (C,H,W). Used after ToTensor in train pipeline."""

    def __init__(
        self,
        p_elastic: float = 0.5,
        p_grid: float = 0.5,
        elastic_sigma: float = 8.0,
        elastic_std: float = 0.08,
        grid_rows: int = 4,
        grid_cols: int = 4,
        grid_std: float = 0.01,
    ):
        self.p_elastic = p_elastic
        self.p_grid = p_grid
        self.elastic_sigma = elastic_sigma
        self.elastic_std = elastic_std
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.grid_std = grid_std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        if random.random() < self.p_elastic:
            x = elastic_deform_tensor(x, self.elastic_std, self.elastic_sigma)
        if random.random() < self.p_grid:
            x = grid_distortion_tensor(x, self.grid_rows, self.grid_cols, self.grid_std)
        if squeeze:
            x = x.squeeze(0)
        return x


# ----- Tensor style ops for TTA (legacy; consistent chain uses PIL + elastic+grid) -----
def _sample_factor(low: float = 0.85, high: float = 1.15) -> float:
    return float(np.random.uniform(low=low, high=high))


def brightness_tensor(x: torch.Tensor, factor: Optional[float] = None) -> torch.Tensor:
    if factor is None:
        factor = _sample_factor(0.85, 1.15)
    return (x * factor).clamp_(x.min().item(), x.max().item())


def contrast_tensor(x: torch.Tensor, factor: Optional[float] = None) -> torch.Tensor:
    if factor is None:
        factor = _sample_factor(0.85, 1.15)
    c = x.shape[1]
    mean = x.view(x.size(0), c, -1).mean(dim=2, keepdim=True).view(x.size(0), c, 1, 1)
    return ((x - mean) * factor + mean).clamp_(x.min().item(), x.max().item())


def color_tensor(x: torch.Tensor, factor: Optional[float] = None) -> torch.Tensor:
    """Saturation-like: blend with grayscale."""
    if factor is None:
        factor = _sample_factor(0.7, 1.3)
    # gray = mean over channels
    gray = x.mean(dim=1, keepdim=True).expand_as(x)
    out = (1 - factor) * gray + factor * x
    return out.clamp_(x.min().item(), x.max().item())


# ImageNet default for denorm/norm in TTA
DEFAULT_NORM_MEAN = [0.485, 0.456, 0.406]
DEFAULT_NORM_STD = [0.229, 0.224, 0.225]


@dataclass
class ImageAugConfig:
    enabled: bool = False
    magnitude: float = 3.0          # severity 1..10 scale for PIL ops
    methods: str = "all"           # "style" | "shape" | "all" (train: full = AugMix 13 + hflip + elastic + grid)
    tta_methods: str = "all"       # same set at test when use_consistent_chain=True
    train_views: int = 0            # extra augmented views per train sample (0 = off)
    test_views: int = 0             # TTA views at test (0 = off)
    chain_depth: int = -1           # -1 = random 1..3
    chain_width: int = 1
    image_size: int = 224
    do_elastic: bool = True
    do_grid: bool = True
    include_flip: bool = True
    elastic_sigma: float = 8.0
    elastic_std: float = 0.08
    grid_rows: int = 4
    grid_cols: int = 4
    grid_std: float = 0.01
    tta_reduce: str = "prob"
    max_views_cap: int = 16
    # Train & test use same full chain (AugMix + hflip + elastic + grid)
    use_consistent_chain: bool = True
    norm_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    norm_std: Tuple[float, ...] = (0.229, 0.224, 0.225)


def _get_tta_style_ops() -> List[Callable]:
    return [
        lambda t: brightness_tensor(t),
        lambda t: contrast_tensor(t),
        lambda t: color_tensor(t),
    ]


def _get_tta_shape_ops(config: ImageAugConfig) -> List[Callable]:
    ops = []
    if config.do_elastic:
        ops.append(lambda t: elastic_deform_tensor(t, config.elastic_std, config.elastic_sigma))
    if config.do_grid:
        ops.append(lambda t: grid_distortion_tensor(t, config.grid_rows, config.grid_cols, config.grid_std))
    if config.include_flip:
        ops.append(hflip_tensor)
    return ops


def _get_pil_ops(methods: str) -> List[Callable]:
    """Return PIL ops for the given method(s). deformation-only returns [] (elastic+grid applied on tensor)."""
    m = (methods or "all").lower()
    if m == "style":
        return list(PIL_OPS_STYLE)
    if m == "shape":
        return list(PIL_OPS_SHAPE)
    if m == "deformation":
        return []
    if m in ("style_shape", "shape_style"):
        return list(PIL_OPS_ALL)
    if m in ("style_deformation", "deformation_style"):
        return list(PIL_OPS_STYLE)
    if m in ("shape_deformation", "deformation_shape"):
        return list(PIL_OPS_SHAPE)
    return list(PIL_OPS_ALL)


def methods_include_deformation(methods: str) -> bool:
    """True if this method set should apply elastic+grid (RevisitingCIL)."""
    m = (methods or "").lower()
    return m in ("deformation", "style_deformation", "deformation_style",
                 "shape_deformation", "deformation_shape", "all")


def apply_pil_chain(
    pil_img: Image.Image,
    severity: float,
    image_size: int,
    ops: List[Callable],
    depth: int = -1,
) -> Image.Image:
    if not ops or depth == 0:
        return pil_img.copy()
    depth = depth if depth > 0 else np.random.randint(1, 4)
    out = pil_img.copy()
    for _ in range(depth):
        op = np.random.choice(ops)
        out = op(out, severity, image_size)
    return out


def build_views_pil(
    pil_img: Image.Image,
    n_views: int,
    config: ImageAugConfig,
) -> List[Image.Image]:
    """Return [original, aug1, aug2, ...] (PIL). Used for training data expansion. deformation-only: same PIL, trsf will add elastic+grid."""
    views = [pil_img.copy()]
    if n_views <= 0 or not config.enabled:
        return views
    ops = _get_pil_ops(config.methods)
    depth = config.chain_depth if config.chain_depth > 0 else -1
    n_views = min(n_views, config.max_views_cap)
    if not ops:
        for _ in range(n_views):
            views.append(pil_img.copy())
        return views
    for _ in range(n_views):
        views.append(apply_pil_chain(pil_img, config.magnitude, config.image_size, ops, depth))
    return views


def _tensor_batch_to_pil_list(
    x: torch.Tensor,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
) -> List[Image.Image]:
    """(B,C,H,W) normalized tensor -> list of B PIL RGB images."""
    mean_t = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    y = x * std_t + mean_t
    y = (y.clamp(0, 1) * 255).byte()
    out = []
    for i in range(y.size(0)):
        arr = y[i].permute(1, 2, 0).cpu().numpy()
        out.append(Image.fromarray(arr))
    return out


def _pil_list_to_tensor_batch(
    pil_list: List[Image.Image],
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    device: torch.device,
    size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """List of B PIL -> (B,C,H,W) normalized tensor. size=(H,W) to resize (e.g. after augment)."""
    tensors = []
    for p in pil_list:
        if size is not None:
            p = p.resize((size[1], size[0]), Image.BILINEAR)
        arr = np.array(p).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        tensors.append(t)
    out = torch.cat(tensors, dim=0).to(device)
    mean_t = torch.tensor(mean, device=device, dtype=out.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=out.dtype).view(1, 3, 1, 1)
    out = (out - mean_t) / std_t
    return out


def build_views_tensor(
    x: torch.Tensor,
    n_views: int,
    config: ImageAugConfig,
) -> List[torch.Tensor]:
    """Return [original, aug1, ...] (tensor). When use_consistent_chain=True, same as train: AugMix 13 + hflip (PIL) + elastic + grid."""
    views = [x]
    if n_views <= 0 or not config.enabled:
        return views
    n_views = min(n_views, config.max_views_cap)
    mean = tuple(config.norm_mean) if hasattr(config.norm_mean, "__iter__") else (0.485, 0.456, 0.406)
    std = tuple(config.norm_std) if hasattr(config.norm_std, "__iter__") else (0.229, 0.224, 0.225)
    depth = config.chain_depth if config.chain_depth > 0 else -1

    if getattr(config, "use_consistent_chain", True):
        # Same chain as train: PIL ops by tta_methods + elastic+grid only when methods include deformation
        ops = _get_pil_ops(config.tta_methods)
        do_def = methods_include_deformation(config.tta_methods) and config.do_elastic and config.do_grid
        h, w = x.size(2), x.size(3)
        for _ in range(n_views):
            pil_list = _tensor_batch_to_pil_list(x, mean, std)
            if ops:
                aug_list = [
                    apply_pil_chain(p, config.magnitude, config.image_size, ops, depth)
                    for p in pil_list
                ]
            else:
                aug_list = pil_list
            v = _pil_list_to_tensor_batch(aug_list, mean, std, x.device, size=(h, w))
            if do_def:
                if config.do_elastic:
                    v = elastic_deform_tensor(v, config.elastic_std, config.elastic_sigma)
                if config.do_grid:
                    v = grid_distortion_tensor(v, config.grid_rows, config.grid_cols, config.grid_std)
            views.append(v)
        return views

    # Legacy: tensor-only style + shape ops
    tta_m = (config.tta_methods or "all").lower()
    style_ops = _get_tta_style_ops() if tta_m in ("style", "all") else []
    shape_ops = _get_tta_shape_ops(config) if tta_m in ("shape", "all") else []
    transforms = style_ops + shape_ops
    if not transforms:
        return views
    for _ in range(n_views):
        v = x
        n_tf = np.random.randint(1, len(transforms) + 1)
        for idx in np.random.permutation(len(transforms))[:n_tf]:
            v = transforms[idx](v)
        views.append(v)
    return views


class ImageAugHelper:
    """Config-driven helper for train data expansion and test-time augmentation."""

    def __init__(self, args: dict):
        norm_mean = args.get("aug_norm_mean")
        if norm_mean is None:
            norm_mean = (0.485, 0.456, 0.406)
        elif isinstance(norm_mean, (list, tuple)) and len(norm_mean) == 3:
            norm_mean = tuple(norm_mean)
        else:
            norm_mean = (0.485, 0.456, 0.406)
        norm_std = args.get("aug_norm_std")
        if norm_std is None:
            norm_std = (0.229, 0.224, 0.225)
        elif isinstance(norm_std, (list, tuple)) and len(norm_std) == 3:
            norm_std = tuple(norm_std)
        else:
            norm_std = (0.229, 0.224, 0.225)
        self.cfg = ImageAugConfig(
            enabled=bool(args.get("aug_enable", False)),
            magnitude=float(args.get("aug_magnitude", 3.0)),
            methods=str(args.get("aug_methods", "all")),
            tta_methods=str(args.get("aug_tta_methods", "all")),
            train_views=int(args.get("aug_train_views", 0)),
            test_views=int(args.get("aug_test_views", 0)),
            chain_depth=int(args.get("aug_chain_depth", -1)),
            chain_width=int(args.get("aug_chain_width", 1)),
            image_size=int(args.get("image_size", 224)),
            do_elastic=bool(args.get("aug_do_elastic", True)),
            do_grid=bool(args.get("aug_do_grid", True)),
            include_flip=bool(args.get("aug_include_flip", True)),
            elastic_sigma=float(args.get("aug_elastic_sigma", 8.0)),
            elastic_std=float(args.get("aug_elastic_std", 0.08)),
            grid_rows=int(args.get("aug_grid_rows", 4)),
            grid_cols=int(args.get("aug_grid_cols", 4)),
            grid_std=float(args.get("aug_grid_std", 0.01)),
            tta_reduce=str(args.get("aug_tta_reduce", "prob")),
            max_views_cap=int(args.get("aug_max_views_cap", 16)),
            use_consistent_chain=bool(args.get("aug_use_consistent_chain", True)),
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    def num_train_views(self) -> int:
        return min(max(0, self.cfg.train_views), self.cfg.max_views_cap) if self.cfg.enabled else 0

    def num_test_views(self) -> int:
        return min(max(0, self.cfg.test_views), self.cfg.max_views_cap) if self.cfg.enabled else 0

    def build_views_pil(self, pil_img: Image.Image) -> List[Image.Image]:
        return build_views_pil(pil_img, self.num_train_views(), self.cfg)

    def build_views_tensor(self, x: torch.Tensor) -> List[torch.Tensor]:
        return build_views_tensor(x, self.num_test_views(), self.cfg)

    def forward_tta(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        device: torch.device,
        output_key: str = "logits",
    ) -> torch.Tensor:
        """TTA: average confidence over original + augmented views; return logits (log-mean-prob)."""
        views = self.build_views_tensor(inputs)
        if len(views) == 1:
            model.eval()
            with torch.no_grad():
                out = model(inputs.to(device))
            logits = out[output_key] if isinstance(out, dict) else out
            return logits

        model.eval()
        all_logits = []
        with torch.no_grad():
            for v in views:
                v = v.to(device)
                out = model(v)
                logits = out[output_key] if isinstance(out, dict) else out
                all_logits.append(logits)
        stacked = torch.stack(all_logits, dim=0)
        if self.cfg.tta_reduce == "logits":
            return stacked.mean(dim=0)
        probs = F.softmax(stacked, dim=-1)
        mean_prob = probs.mean(dim=0)
        return torch.log(mean_prob + 1e-8)


def _default_pil_loader(path: str) -> Image.Image:
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    with open(path, "rb") as f:
        try:
            img = Image.open(f)
            return img.convert("RGB")
        except OSError:
            f.seek(0)
            img = Image.open(f)
            img.load()
            return img.convert("RGB")


class AugmentedDataset(torch.utils.data.Dataset):
    """Wraps a path/array dataset to add augmented views per sample (train set expansion)."""

    def __init__(
        self,
        images,
        labels,
        trsf,
        use_path: bool = True,
        aug_helper: Optional[ImageAugHelper] = None,
        pil_loader: Optional[Callable] = None,
    ):
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path
        self.aug_helper = aug_helper
        self.pil_loader = pil_loader or _default_pil_loader
        self.n_base = len(images)
        self.n_views = 1 + (aug_helper.num_train_views() if aug_helper and aug_helper.enabled else 0)
        self._length = self.n_base * self.n_views

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int):
        base_idx = idx // self.n_views
        view_idx = idx % self.n_views
        label = int(self.labels[base_idx])

        if self.use_path:
            pil_img = self.pil_loader(self.images[base_idx])
        else:
            pil_img = Image.fromarray(self.images[base_idx])

        if view_idx > 0 and self.aug_helper and self.aug_helper.enabled:
            views = build_views_pil(pil_img, self.aug_helper.num_train_views(), self.aug_helper.cfg)
            if view_idx <= len(views) - 1:
                pil_img = views[view_idx]

        image = self.trsf(pil_img)
        return idx, image, label
