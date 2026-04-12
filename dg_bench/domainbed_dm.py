"""
DomainBed-style data manager for CIDG (Class-Incremental Domain Generalization).
Leave-one-domain-out: source domains split 80% train / 20% val; test = held-out domain.
Compatible with C3Box: provides get_dataset(source="train"|"val"|"test"), _class_to_label, _data_to_prompt for CLIP.
"""
import os
import logging
import numpy as np
from PIL import Image, ImageFile
from typing import List, Tuple, Dict

from torch.utils.data import Dataset
from torchvision import transforms


# Default CLIP prompt for DG datasets (no dataset-specific labels file)
DEFAULT_DG_TEMPLATES = ["a photo of a {}."]

# DomainBed dataset name -> folder name under dataset_root
DATASET_DIR_MAP = {
    "PACS": "PACS",
    "VLCS": "VLCS",
    "OfficeHome": "office_home",
    "terra_incognita": "terra_incognita",
    "domain_net": "domain_net",
}


def _dataset_dir(dataset_root: str, dataset_name: str) -> str:
    folder = DATASET_DIR_MAP.get(dataset_name, dataset_name)
    return os.path.join(dataset_root, folder)


class DomainBedDataManager(object):

    def __init__(
        self,
        dataset_root: str,
        dataset_name: str,
        train_domains: List[str],
        test_domain: str,
        shuffle: bool,
        seed: int,
        init_cls: int,
        increment: int,
        image_size: int = 224,
        val_split_ratio: float = 0.2,  # DomainBed: 20% of source domains for validation
    ) -> None:
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name
        self.train_domains = list(train_domains)
        self.test_domain = test_domain
        self.val_split_ratio = val_split_ratio

        self._setup_data(shuffle=shuffle, seed=seed, image_size=image_size)

        assert init_cls <= len(self._class_order), "No enough classes."
        # avoid creating a zero-sized first task when init_cls == 0
        self._increments = [] if init_cls == 0 else [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)
        # clg_cbm compatibility (ensure concept_order exists even if _setup_data is from older code)
        self.concept_order = list(range(len(self._class_order)))

    @property
    def nb_tasks(self) -> int:
        return len(self._increments)

    def get_task_size(self, task: int) -> int:
        return self._increments[task]

    def get_accumulate_tasksize(self, task: int) -> int:
        return sum(self._increments[: task + 1])

    def get_total_classnum(self) -> int:
        return len(self._class_order)

    def get_task_class_indices(self, task: int):
        start = sum(self._increments[:task]) if task > 0 else 0
        size = self._increments[task]
        return list(range(start, start + size))

    def get_class_name_by_new_index(self, new_index: int) -> str:
        orig_id = self._class_order[new_index]
        return self._classes[orig_id]

    def get_class_counts(self, source: str):
        if source == "train":
            y = self._train_targets
        elif source == "val":
            y = self._val_targets
        elif source == "test":
            y = self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))
        counts = {}
        for i in range(len(self._class_order)):
            counts[i] = int((y == i).sum())
        return counts

    def getlen(self, index: int) -> int:
        """Return number of training samples for class index (for FOSTER etc.)."""
        return int((self._train_targets == index).sum())

    def get_attributes(self, attribute: str, indice: List[int]):
        """CLG-CBM: no concept files for DG datasets; use class names as single attribute per class."""
        names = [self.get_class_name_by_new_index(i) for i in indice]
        attr = list(names)
        cpt_count = [0] + list(range(1, len(indice) + 1))
        return attr, names, cpt_count

    def get_prefix(self, args) -> str:
        """CLG-CBM: generic prefix for DG datasets."""
        name = (args.get("dataset", self.dataset_name) if isinstance(args, dict) else self.dataset_name) or ""
        name = name.lower()
        if "cifar" in name:
            return "A bad photo of an object with "
        if "cub" in name:
            return "The bird has "
        return "A photo of an object with "

    def get_dataset(
        self,
        indices,
        source: str,
        mode: str,
        appendent=None,
        ret_data: bool = False,
        m_rate=None,
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "val":
            x, y = self._val_data, self._val_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
            # Insert elastic+grid only when methods include "deformation" (风格/形状/形变 组合)
            if getattr(self, "aug_helper", None) and self.aug_helper.enabled:
                from utils.image_aug import ElasticGridTransform, methods_include_deformation
                c = self.aug_helper.cfg
                if methods_include_deformation(c.methods):
                    trsf = transforms.Compose([
                        *self._train_trsf,
                        ElasticGridTransform(
                            p_elastic=0.5, p_grid=0.5,
                            elastic_sigma=c.elastic_sigma, elastic_std=c.elastic_std,
                            grid_rows=c.grid_rows, grid_cols=c.grid_cols, grid_std=c.grid_std,
                        ),
                        *self._common_trsf,
                    ])
        elif mode == "flip":
            trsf = transforms.Compose([
                *self._test_trsf,
                transforms.RandomHorizontalFlip(p=1.0),
                *self._common_trsf,
            ])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx + 1)
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        if len(data) == 0:
            data_arr = np.array([])
            targets_arr = np.array([])
        else:
            data_arr, targets_arr = np.concatenate(data), np.concatenate(targets)

        dataset = _DummyDataset(data_arr, targets_arr, trsf, use_path=True)
        if source == "train" and getattr(self, "aug_helper", None) and self.aug_helper.num_train_views() > 0:
            from utils.image_aug import AugmentedDataset
            dataset = AugmentedDataset(
                data_arr, targets_arr, trsf, use_path=True,
                aug_helper=self.aug_helper, pil_loader=_pil_loader,
            )
        if ret_data:
            return data_arr, targets_arr, dataset
        return dataset

    def _setup_data(self, shuffle: bool, seed: int, image_size: int) -> None:
        train_paths, train_labels, class_to_idx = self._gather_domains(self.train_domains)
        test_paths, test_labels, _ = self._gather_domains([self.test_domain], class_to_idx)

        # DomainBed protocol: split source domains into train/val
        np.random.seed(seed)
        n_samples = len(train_paths)
        indices = np.random.permutation(n_samples)
        val_size = int(n_samples * self.val_split_ratio)

        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        # Split source domain data into train and validation
        self._train_data = np.array([train_paths[i] for i in train_indices])
        self._train_targets = np.array([train_labels[i] for i in train_indices])
        self._val_data = np.array([train_paths[i] for i in val_indices])
        self._val_targets = np.array([train_labels[i] for i in val_indices])

        # Test domain remains unchanged
        self._test_data = np.array(test_paths)
        self._test_targets = np.array(test_labels)
        self.use_path = True

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self._train_trsf = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        self._test_trsf = [
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
        self._common_trsf = [normalize]

        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = list(range(len(self._classes)))
        self._class_order = order
        logging.info(self._class_order)

        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._val_targets = _map_new_class_index(self._val_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

        # C3Box CLIP compatibility: class names and templates (order by new index)
        self._class_to_label = [self._classes[i] for i in self._class_order]
        self._data_to_prompt = list(DEFAULT_DG_TEMPLATES)
        # clg_cbm compatibility: concept order = class order (new indices)
        self.concept_order = list(range(len(self._class_order)))

    def _gather_domains(
        self, domains: List[str], existing_class_to_idx: Dict[str, int] = None
    ) -> Tuple[List[str], List[int], Dict[str, int]]:
        dataset_dir = _dataset_dir(self.dataset_root, self.dataset_name)
        paths: List[str] = []
        labels: List[int] = []
        if existing_class_to_idx is None:
            class_to_idx: Dict[str, int] = {}
        else:
            class_to_idx = dict(existing_class_to_idx)

        for domain in domains:
            domain_dir = os.path.join(dataset_dir, domain)
            if not os.path.isdir(domain_dir):
                raise FileNotFoundError("Domain directory not found: {}".format(domain_dir))
            for cls in sorted(os.listdir(domain_dir)):
                cls_dir = os.path.join(domain_dir, cls)
                if not os.path.isdir(cls_dir):
                    continue
                if cls not in class_to_idx:
                    class_to_idx[cls] = len(class_to_idx)
                cls_idx = class_to_idx[cls]
                for root, _, files in os.walk(cls_dir):
                    for f in files:
                        lf = f.lower()
                        if lf.endswith(".jpg") or lf.endswith(".jpeg") or lf.endswith(".png") or lf.endswith(".bmp"):
                            paths.append(os.path.join(root, f))
                            labels.append(cls_idx)

        inv = [None] * len(class_to_idx)
        for k, v in class_to_idx.items():
            inv[v] = k
        self._classes = inv

        return paths, labels, class_to_idx

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]


class _DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=True):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(_pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = int(self.labels[idx])
        return idx, image, label


def _pil_loader(path):
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


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))
