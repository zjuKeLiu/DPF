from functools import partial

# from pathlib import Path
from typing import Any, Dict, Union

import ignite
import torch
from accelerate import Accelerator
# from accelerate import set_seed
from accelerate.logging import get_logger
from ignite.contrib.handlers import TensorboardLogger
try:
    from ignite.contrib.handlers.stores import EpochOutputStore
except Exception as exp:
    from ignite.handlers.stores import EpochOutputStore

    pass
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers.tensorboard_logger import (
    global_step_from_engine,
)
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import (
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.contrib.metrics import ROC_AUC, RocCurve
from ignite.metrics import (
    Accuracy,
    Precision,
    Recall,
    ConfusionMatrix,
)
import pickle as pk
import numpy as np
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import Loss, MeanAbsoluteError
from torch import nn
from data import get_train_val_loaders
from config import TrainingConfig
from models.pyg_att import Matformer

from jarvis.db.jsonutils import dumpjson
import json
import pprint

import os
import warnings
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
import time

# torch config
torch.set_default_dtype(torch.float32)

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")



class CustomMetric_lattice(Metric):
    def __init__(self, output_transform=lambda x: x):
        self._samplenum = []
        self._mae = []
        self._MAE_loss = nn.L1Loss()
        super(CustomMetric_lattice, self).__init__(output_transform=output_transform)
    def reset(self):
        self._samplenum = []
        self._mae = []
        
    def update(self, output):
        y_pred, y_gt = output
        self._samplenum.append(y_pred["lattice"].shape[0])
        self._mae.append(self._MAE_loss(y_pred["lattice"], y_gt["lattice"].t().view(-1,3,3)).item())

    def compute(self):
        return sum(w*v for w, v in zip(self._mae, self._samplenum)) / sum(self._samplenum)


class CustomMetric_position(Metric):
    def __init__(self, output_transform=lambda x: x):
        self._samplenum = []
        self._mae = []
        self._MAE_loss = nn.L1Loss()
        super(CustomMetric_position, self).__init__(output_transform=output_transform)
        
    def reset(self):
        self._samplenum = []
        self._mae = []
        
    def update(self, output):
        y_pred, y_gt = output
        self._samplenum.append(y_pred["positions"].shape[0])
        self._mae.append(self._MAE_loss(y_pred["positions"], y_gt["positions"].t()).item())

    def compute(self):
        return sum(w*v for w, v in zip(self._mae, self._samplenum)) / sum(self._samplenum)

class CustomMetric_atom(Metric):
    def __init__(self, output_transform=lambda x: x):
        self._num_examples = 0
        self._correct =0
        super(CustomMetric_atom, self).__init__(output_transform=output_transform)

    def reset(self):
        self._correct = 0
        self._num_examples = 0

    def update(self, output):
        y_pred_dict, y_gt_dict = output
        y_pred = y_pred_dict["atoms"]
        label = y_gt_dict["atoms"]
        mask = y_gt_dict["mask"]
        _, y_pred_class = torch.max(y_pred, 1)
        #predd = torch.masked_select(y_pred_class, mask==1)
        #targett = torch.masked_select(label, mask==1)
        #self._correct += (predd == targett).sum().item()
        #self._num_examples += len(targett)
        self._correct += ((y_pred_class==label)*mask).sum().item()
        self._num_examples += mask.sum().item()
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CustomMetric must have at least one example before it can be computed.')
        return self._correct / self._num_examples


def activated_output_transform(output):
    """Exponentiate output."""
    y_pred, y = output
    y_pred = torch.exp(y_pred)
    y_pred = y_pred[:, 1]
    return y_pred, y


def make_standard_scalar_and_pca(output):
    """Use standard scalar and PCS for multi-output data."""
    sc = pk.load(open(os.path.join(tmp_output_dir, "sc.pkl"), "rb"))
    y_pred, y = output
    y_pred = torch.tensor(sc.transform(y_pred.cpu().numpy()), device=device)
    y = torch.tensor(sc.transform(y.cpu().numpy()), device=device)
    return y_pred, y


def thresholded_output_transform(output):
    """Round off output."""
    y_pred, y = output
    y_pred = torch.round(torch.exp(y_pred))
    # print ('output',y_pred)
    return y_pred, y


def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]


def setup_optimizer(params, config: TrainingConfig):
    """Set up optimizer for param groups."""
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    return optimizer


def train_dgl(
    config: Union[TrainingConfig, Dict[str, Any]],
    model: nn.Module = None,
    train_val_test_loaders=[],
    test_only=False,
    use_save=True,
    mp_id_list=None,
    pre_train = None,
):
    """
    `config` should conform to matformer.conf.TrainingConfig, and
    if passed as a dict with matching keys, pydantic validation is used
    """
    from accelerate import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    # init the accelerator
    #accelerator = Accelerator()
    device=accelerator.device

    if accelerator.is_local_main_process:
        print(config)
    if type(config) is dict:
        try:
            config = TrainingConfig(**config)
        except Exception as exp:
            print("Check", exp)
            print('error in converting to training config!')
    import os
    
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    checkpoint_dir = os.path.join(config.output_dir)
    deterministic = False
    classification = False
    print("config:")
    tmp = config.dict()
    f = open(os.path.join(config.output_dir, "config.json"), "w")
    f.write(json.dumps(tmp, indent=4))
    f.close()
    global tmp_output_dir
    tmp_output_dir = config.output_dir
    pprint.pprint(tmp) 
    if config.classification_threshold is not None:
        classification = True
    if config.random_seed is not None:
        deterministic = True
        ignite.utils.manual_seed(config.random_seed)
    
    std_train = None
    mean_train = None
    line_graph = True
    train_loader = train_val_test_loaders[0]
    val_loader = train_val_test_loaders[1]
    test_loader = train_val_test_loaders[2]
    prepare_batch = train_val_test_loaders[3]
    prepare_batch = partial(prepare_batch, device=device)
    if classification:
        config.model.classification = True
    # define network, optimizer, scheduler
    _model = {
        "matformer" : Matformer,
    }
    if std_train is None:
        std_train = 1.0
    if model is None:
        net = _model.get(config.model.name)(config.model)
        print("config:")
        pprint.pprint(config.model.dict())
    else:
        net = model

    net.to(device)
    params = group_decay(net)
    optimizer = setup_optimizer(params, config)

    if config.scheduler == "none":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1.0
        )

    elif config.scheduler == "onecycle":
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
        )
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=100000,
            gamma=0.96,
        )

    # select configured loss function
    criteria = {
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
    }
    criterion = criteria[config.criterion]
    # set up training engine and evaluators
    metrics = {"loss": Loss(criterion), "mae": MeanAbsoluteError() * std_train, "neg_mae": -1.0 * MeanAbsoluteError() * std_train}
    if pre_train:
        class Criterion(nn.Module):
            def __init__(self, mask_ratio, position_noise, lattice_noise):
                super(Criterion, self).__init__()
                self.ce = nn.CrossEntropyLoss(reduction='none')
                self.l1 = nn.L1Loss()
                self._samplenum = []
                self._mae = []
                self.mask_ratio = (mask_ratio is None)
                self.position_noise = (position_noise is None)
                self.lattice_noise = (lattice_noise is None)

                #self.return_keys = []
                #self.loss_func = {"atoms":self.loss_atoms, "positon": self.l1, "lattice": self.l1}
                #if mask_ratio is not None:
                #    self.return_keys.append("atoms")
                #if position_noise is not None:
                #    self.return_keys.append("position")
                #if lattice_noise is not None:
                #    self.return_keys.append("lattice")
                self.loss_step = []
            def loss_atoms(self, label_pred, label, mask):
                ce_loss_items = self.ce(label_pred, label)
                #selected_loss = torch.masked_select(ce_loss_items, mask==1)
                #mean_loss = torch.mean(selected_loss)
                mean_loss = (ce_loss_items*mask).sum()/mask.sum()
                return mean_loss

            def reset(self):
                self._samplenum = []
                self._mae = []
            
            def update(self, output):
                y_pred, y_gt = output
                for k, value in y_pred.items():
                    _samplenum = value.shape[0]
                    break
                self._samplenum.append(_samplenum)
                self._mae.append(self.forward(y_pred, y_gt).item())
            
            def compute(self):
                return sum(w*v for w, v in zip(self._mae, self._samplenum)) / sum(self._samplenum)
                       
            def forward(self, y_pred, y_gt):
                all_loss = 0

                if "atoms" in y_pred.keys():
                    atom_loss = self.loss_atoms(y_pred["atoms"], y_gt["atoms"], y_gt["mask"])
                    all_loss += atom_loss
                if "positions" in y_pred.keys():
                    position_loss = self.l1(y_pred["positions"], y_gt["positions"].t())
                    all_loss += position_loss
                if "lattice" in y_pred.keys():
                    lattice_loss = self.l1(y_pred["lattice"], y_gt["lattice"].t().view(-1,3,3))
                    all_loss += lattice_loss
                '''
                inx = 0
                atom_loss = 0
                position_loss = 0
                lattice_loss = 0
                if self.mask_ratio:
                    atom_loss = self.loss_atoms(y_pred[inx], y_gt["atoms"], y_gt["mask"])
                    inx += 1
                if self.position_noise:
                    position_loss = self.l1(y_pred[inx], y_gt["positions"].t())
                    all_loss += position_loss
                    inx += 1
                if self.lattice_noise:
                    lattice_loss = self.l1(y_pred[inx], y_gt["lattice"].t().view(-1,3,3))
                    all_loss += lattice_loss
                all_loss = atom_loss + position_loss + lattice_loss
                '''
                return all_loss

        criterion = Criterion(config.mask_ratio, config.position_noise, config.lattice_noise)
        
        if accelerator.is_local_main_process:
            metrics = {"loss": criterion}
            history_dict_train = {}
            history_dict_val = {}
            history_dict_test = {}
            history_dict_train["loss"] = []
            history_dict_val["loss"] = []
            history_dict_test["loss"] = []
            if config.model.mask_ratio is not None:
                metrics["atom_acc"] = CustomMetric_atom()
                history_dict_train["atom_acc"] = []
                history_dict_val["atom_acc"] = []
                history_dict_test["atom_acc"] = []
            if config.model.position_noise is not None:
                metrics["position_mae"] = CustomMetric_position()
                history_dict_train["position_mae"] = []
                history_dict_val["position_mae"] = []
                history_dict_test["position_mae"] = []
            if config.model.lattice_noise is not None:
                metrics["lattice_mae"] = CustomMetric_lattice()
                history_dict_train["lattice_mae"] = []
                history_dict_val["lattice_mae"] = []
                history_dict_test["lattice_mae"] = []
            store_list = []
            for i in range(5):
                store = config.epochs-10*i
                if store > 0:
                    store_list.append(store)
                else:
                    break

        best_loss = np.inf
        prepare_batch,net,optimizer,train_loader,val_loader,test_loader,scheduler = \
            accelerator.prepare(prepare_batch,net,optimizer,train_loader,val_loader,test_loader,scheduler)

        if accelerator.is_local_main_process:
            t1 = time.time()
        for e in range(config.epochs):
            for inx, data in enumerate(train_loader):
                optimizer.zero_grad()
                results = net([data[0], data[1]])
                loss = criterion.forward(results, data[2])
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
            with torch.no_grad():        
                if e%10 == 0:
                    for train_data in train_loader:
                        return_dict = net((train_data[0], train_data[1]))
                        if accelerator.is_local_main_process:        
                            for kk in metrics:
                                metrics[kk].update((return_dict, train_data[2]))
                    if accelerator.is_local_main_process:
                        for kk in metrics:
                            history_dict_train[kk].append(metrics[kk].compute())
                            metrics[kk].reset()
                    optimizer.zero_grad()
                    
                for eval_data in val_loader:
                    return_dict = net((eval_data[0], eval_data[1]))
                    if accelerator.is_local_main_process:
                        for kk in metrics:
                            metrics[kk].update((return_dict, eval_data[2]))
                        print_str = []
                    optimizer.zero_grad()
                if accelerator.is_local_main_process:
                    for kk in metrics:
                        history_dict_val[kk].append(metrics[kk].compute())
                        metrics[kk].reset()
                        print_str.append(kk)
                        print_str.append(str(history_dict_val[kk][-1]))
                    my_string = ' '.join(print_str)
                    print(f'{e+1}/{config.epochs}:{my_string}')
                    if best_loss > history_dict_val["loss"][-1]:
                        best_loss = history_dict_val["loss"][-1]
                        unwrap_model=accelerator.unwrap_model(net)
                        torch.save(
                            unwrap_model.state_dict(),
                            os.path.join(config.output_dir, f'best.pt'),
                        )
                    if e in store_list:
                        torch.save(
                            unwrap_model.state_dict(),
                            os.path.join(config.output_dir, f'model_{e}.pt'),
                        )
        
        for test_data in test_loader:
            optimizer.zero_grad()
            with torch.no_grad():
                return_dict = net([test_data[0], test_data[1]])
                if accelerator.is_local_main_process:
                    for kk in metrics:
                        metrics[kk].update([return_dict, test_data[2]])
                    for kk in metrics:
                        history_dict_test[kk].append(metrics[kk].compute())

        if accelerator.is_local_main_process:
            t2 = time.time()
            print("Total time:", t2-t1)
            history_dict_test["time"] = t2-t1
            dumpjson(
                filename=os.path.join(config.output_dir, "history_dict_test.json"),
                data=history_dict_test,
            )
            dumpjson(
                filename=os.path.join(config.output_dir, "history_dict_val.json"),
                data=history_dict_val,
            )
            dumpjson(
                filename=os.path.join(config.output_dir, "history_dict_train.json"),
                data=history_dict_train,
            )