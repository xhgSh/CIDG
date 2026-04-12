import logging
import numpy as np
import torch
from copy import deepcopy
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.optim import Optimizer
import math
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import CodaPromptVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy

# tune the model at first session with vpt, and then conduct simple shot.
num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        self._network = CodaPromptVitNet(args, True)

        self.batch_size = 128
        self.init_lr = 0.001
        self.weight_decay = 0
        self.min_lr =  1e-6
        self.args = args

        total_params = sum(p.numel() for p in self._network.parameters())
        logging.info(f'{total_params:,} total parameters.')
        for p in self._network.backbone.parameters():
            p.requires_grad_(False)
        total_trainable_params = sum(p.numel() for p in self._network.fc.parameters() if p.requires_grad) + sum(
            p.numel() for p in self._network.prompt.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} fc and prompt training parameters.')

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1

        if self._cur_task > 0:
            try:
                if self._network.module.prompt is not None:
                    self._network.module.prompt.process_task_count()
            except:
                if self._network.prompt is not None:
                    self._network.prompt.process_task_count()

        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        # self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train", appendent=self._get_memory())
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True,
                                       num_workers=num_workers)
        self.val_loader = self._create_val_loader_if_available(data_manager)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False,
                                      num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        eval_loader = self.val_loader if self.val_loader is not None else self.test_loader
        self._train(self.train_loader, eval_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self.build_rehearsal_memory(data_manager, self.samples_per_class)

    def _train(self, train_loader, eval_loader):
        self._network.to(self._device)

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        self.data_weighting()
        self._init_train(train_loader, eval_loader, optimizer, scheduler)

    def data_weighting(self):
        self.dw_k = torch.tensor(np.ones(self._total_classes + 1, dtype=np.float32))
        self.dw_k = self.dw_k.to(self._device)

    def get_optimizer(self):
        if len(self._multiple_gpus) > 1:
            params = list(self._network.module.prompt.parameters()) + list(self._network.module.fc.parameters())
        else:
            params = list(self._network.prompt.parameters()) + list(self._network.fc.parameters())
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(params, momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(params, lr=self.init_lr, weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)

        return optimizer

    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = CosineSchedule(optimizer, K=self.args["tuned_epoch"])
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"],
                                                       gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_train(self, train_loader, eval_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        best_val_acc = 0.0
        best_model_state = None
        eval_name = "Val" if self.val_loader is not None else "Test"
        for _, epoch in enumerate(prog_bar):
            self._network.train()

            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                # logits
                logits, prompt_loss = self._network(inputs, train=True)
                logits = logits[:, :self._total_classes]

             #   logits[:, :self._known_classes] = float('-inf')
                dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
                loss_supervised = (F.cross_entropy(logits, targets.long()) * dw_cls).mean()

                # ce loss
                loss = loss_supervised + prompt_loss.sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            eval_acc = self._compute_accuracy(self._network, eval_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, {}_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
                eval_name,
                eval_acc,
            )
            if eval_acc > best_val_acc:
                best_val_acc = eval_acc
                best_model_state = deepcopy(self._network.module.state_dict() if isinstance(self._network, nn.DataParallel) else self._network.state_dict())

            # if (epoch + 1) % 5 == 0:
            #
            # else:
            #     info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
            #         self._cur_task,
            #         epoch + 1,
            #         self.args['tuned_epoch'],
            #         losses / len(train_loader),
            #         train_acc,
            #     )
            prog_bar.set_description(info)

        if best_model_state is not None:
            (self._network.module if isinstance(self._network, nn.DataParallel) else self._network).load_state_dict(best_model_state)
            logging.info("Task {}: Restored best checkpoint with {}_acc {:.2f}%".format(self._cur_task, eval_name, best_val_acc))
        logging.info(info)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)[:, :self._total_classes]
            k_eff = min(self.topk, outputs.size(1))
            predicts = torch.topk(outputs, k=k_eff, dim=1, largest=True, sorted=True)[1]
            if k_eff < self.topk:
                pad = predicts[:, :1].expand(-1, self.topk - k_eff)
                predicts = torch.cat([predicts, pad], dim=1)
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)[:, :self._total_classes]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _get_eval_logits_for_fusion(self):
        """
        为 BaseLearner.eval_task 的 logits 融合提供专用接口（CodaPrompt 专用）。
        注意：这里必须与本类自定义的 _eval_cnn 使用完全一致的前向路径
        （self._network(inputs)[:, :self._total_classes]），
        否则 BaseLearner 中默认的 self._network(inputs)["logits"] 会报
        “too many indices for tensor of dimension 2”，并导致融合直接失败。
        """
        self._network.eval()
        logits_list, targets_list = [], []
        for _, (_, inputs, targets) in enumerate(self.test_loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)[:, :self._total_classes]
            logits_list.append(outputs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

        import numpy as _np
        cil_logits = _np.concatenate(logits_list, axis=0)
        cil_targets = _np.concatenate(targets_list, axis=0)
        return cil_logits, cil_targets


class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class CosineSchedule(_LRScheduler):

    def __init__(self, optimizer, K):
        self.K = K
        super().__init__(optimizer, -1)

    def cosine(self, base_lr):
        return base_lr * math.cos((99 * math.pi * (self.last_epoch)) / (200 * (self.K - 1)))

    def get_lr(self):
        return [self.cosine(base_lr) for base_lr in self.base_lrs]