import logging
import numpy as np
import torch
from copy import deepcopy
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import DualpromptVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy

# tune the model at first session with vpt, and then conduct simple shot.
num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        self._network = DualpromptVitNet(args, True)

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args

        # Freeze the parameters for ViT.
        if self.args["freeze"]:
            for p in self._network.original_backbone.parameters():
                p.requires_grad = False
            for p in self._network.backbone.parameters():
                p.requires_grad = False
        self._network.backbone.e_prompt.requires_grad_(True)
        self._network.backbone.head.requires_grad_(True)


        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        logging.info(f'{total_params:,} model total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} model training parameters.')

        # if some parameters are trainable, print the key name and corresponding parameter number
        if total_params != total_trainable_params:
            for name, param in self._network.backbone.named_parameters():
                if param.requires_grad:
                    logging.info("{}: {}".format(name, param.numel()))

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        # self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train", appendent=self._get_memory())
        # train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
        #                                          mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = self._create_val_loader_if_available(data_manager)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        eval_loader = self.val_loader if self.val_loader is not None else self.test_loader
        self._train(self.train_loader, eval_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self.build_rehearsal_memory(data_manager, self.samples_per_class)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        if self._cur_task > 0:
            self._init_prompt(optimizer)

        if self._cur_task > 0 and self.args["reinit_optimizer"]:
            optimizer = self.get_optimizer()

        self._init_train(train_loader, test_loader, optimizer, scheduler)

    def get_optimizer(self):
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=self.init_lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.init_lr,
                weight_decay=self.weight_decay
            )

        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.init_lr,
                weight_decay=self.weight_decay
            )

        return optimizer

    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args['tuned_epoch'],
                                                             eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"],
                                                       gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_prompt(self, optimizer):
        args = self.args
        model = self._network.backbone
        task_id = self._cur_task

        # Transfer previous learned prompt params to the new prompt
        if args["prompt_pool"] and args["shared_prompt_pool"]:
            prev_start = (task_id - 1) * args["top_k"]
            prev_end = task_id * args["top_k"]

            cur_start = prev_end
            cur_end = (task_id + 1) * args["top_k"]

            if (prev_end > args["size"]) or (cur_end > args["size"]):
                pass
            else:
                cur_idx = (slice(None), slice(None), slice(cur_start, cur_end)) if args[
                    "use_prefix_tune_for_e_prompt"] else (slice(None), slice(cur_start, cur_end))
                prev_idx = (slice(None), slice(None), slice(prev_start, prev_end)) if args[
                    "use_prefix_tune_for_e_prompt"] else (slice(None), slice(prev_start, prev_end))

                with torch.no_grad():
                    model.e_prompt.prompt.grad.zero_()
                    model.e_prompt.prompt[cur_idx] = model.e_prompt.prompt[prev_idx]
                    optimizer.param_groups[0]['params'] = model.parameters()

        # Transfer previous learned prompt param keys to the new prompt
        if args["prompt_pool"] and args["shared_prompt_key"]:
            prev_start = (task_id - 1) * args["top_k"]
            prev_end = task_id * args["top_k"]

            cur_start = prev_end
            cur_end = (task_id + 1) * args["top_k"]

            if (prev_end > args["size"]) or (cur_end > args["size"]):
                pass
            else:
                cur_idx = (slice(cur_start, cur_end))
                prev_idx = (slice(prev_start, prev_end))

            with torch.no_grad():
                model.e_prompt.prompt_key.grad.zero_()
                model.e_prompt.prompt_key[cur_idx] = model.e_prompt.prompt_key[prev_idx]
                optimizer.param_groups[0]['params'] = model.parameters()

    def _init_train(self, train_loader, eval_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        best_val_acc = 0.0
        best_model_state = None
        eval_name = "Val" if self.val_loader is not None else "Test"
        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            self._network.original_backbone.eval()

            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                output = self._network(inputs, task_id=self._cur_task, train=True)
                logits = output["logits"][:, :self._total_classes]
              #  logits[:, :self._known_classes] = float('-inf')

                loss = F.cross_entropy(logits, targets.long())
                if self.args["pull_constraint"] and 'reduce_sim' in output:
                    loss = loss - self.args["pull_constraint_coeff"] * output['reduce_sim']

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
                outputs = self._network(inputs, task_id=self._cur_task)["logits"][:, :self._total_classes]
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
                outputs = model(inputs, task_id=self._cur_task)["logits"][:, :self._total_classes]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _get_eval_logits_for_fusion(self):
        """
        为 BaseLearner.eval_task 的 logits 融合提供专用接口（DualPrompt 专用）。
        这里必须与本类自定义的 _eval_cnn / _compute_accuracy 使用同一条前向路径
        （self._network(inputs, task_id=self._cur_task)["logits"][:, :self._total_classes]），
        否则 BaseLearner 中默认的 self._network(inputs)["logits"] 会忽略 task_id，
        导致 logits 形状或语义不一致，从而使融合结果失效或报错。
        """
        self._network.eval()
        logits_list, targets_list = [], []
        for _, (_, inputs, targets) in enumerate(self.test_loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs, task_id=self._cur_task)["logits"][:, :self._total_classes]
            logits_list.append(outputs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

        import numpy as _np
        cil_logits = _np.concatenate(logits_list, axis=0)
        cil_targets = _np.concatenate(targets_list, axis=0)
        return cil_logits, cil_targets