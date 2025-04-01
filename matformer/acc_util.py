import os
import csv
from jarvis.core.atoms import Atoms
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import time
from pandarallel import pandarallel

def one_poscar_data(file_name):
    #root_dir = "/data/llm_pre_train_poscar"
    root_dir = "/mnt/nas/share2/home/liuke/data/llm_pre_train_poscar"
    # root_dir = "/mnt/nas/share2/home/liuke/data/CrystalData/benchmark/shear_megnet"
    file_path = os.path.join(root_dir, file_name)
    #atoms = Atoms.from_poscar((file_path+".vasp"))
    atoms = Atoms.from_poscar(file_path)
    info = {}
    info["atoms"] = atoms.to_dict()
    info["jid"] = file_name
    info["target"] = float(0.0)
    return info


def get_data(root_dir, file_format='poscar', debug=False):
    print("#############get data##################")
    t1 = time.time()
    pandarallel.initialize(nb_workers=32)
    id_prop_dat = os.path.join(root_dir, "id_prop.csv")
    df = pd.read_csv(id_prop_dat, header=None)
    dataset = df[0].parallel_apply(one_poscar_data).values
    n_outputs = df[1].values
    t2 = time.time()
    print(f"#############get data done {t2-t1} s##################")
    return list(dataset), list(n_outputs)

    '''
    with open(id_prop_dat, "r") as f:
        reader = csv.reader(f)
        data = [row for row in reader]

    dataset = []
    n_outputs = []
    
    for ind, i in tqdm(enumerate(data)):
        if debug and ind>200:
            break
        info = {}
        file_name = i[0]
        file_path = os.path.join(root_dir, file_name)
        if file_format == "poscar":
            atoms = Atoms.from_poscar(file_path)
        elif file_format == "cif" or file_format == "CIF":
            atoms = Atoms.from_cif(file_path)
        elif file_format == "xyz":
            # Note using 500 angstrom as box size
            atoms = Atoms.from_xyz(file_path, box_size=500)
        elif file_format == "pdb":
            # Note using 500 angstrom as box size
            # Recommended install pytraj
            # conda install -c ambermd pytraj
            atoms = Atoms.from_pdb(file_path, max_lat=500)
        else:
            raise NotImplementedError(
                "File format not implemented", file_format
            )

        info["atoms"] = atoms.to_dict()
        info["jid"] = file_name

        tmp = [float(j) for j in i[1:]]  # float(i[1])
        if len(tmp) == 1:
            tmp = tmp[0]
        info["target"] = tmp  # float(i[1])
        n_outputs.append(info["target"])
        dataset.append(info)
    return dataset, n_outputs
    '''

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
        self.loss_step = []
    def loss_atoms(self, label_pred, label, mask):
        ce_loss_items = self.ce(label_pred, label)
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
            #print(y_pred["atoms"].shape, y_gt["atoms"].shape, y_gt["mask"].shape)
            atom_loss = self.loss_atoms(y_pred["atoms"], y_gt["atoms"], y_gt["mask"])
            all_loss += atom_loss
        if "positions" in y_pred.keys():
            position_loss = self.l1(y_pred["positions"], y_gt["positions"].t())
            all_loss += position_loss
        if "lattice" in y_pred.keys():
            lattice_loss = self.l1(y_pred["lattice"], y_gt["lattice"].t().view(-1,3,3))
            all_loss += lattice_loss
        return all_loss