"""Implementation based on the template of ALIGNN."""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pydantic.typing import Literal
from torch import nn
from .utils import RBFExpansion
from features import angle_emb_mp
from torch_scatter import scatter
from .transformer import MatformerConv

from pydantic import BaseSettings as PydanticBaseSettings


class BaseSettings(PydanticBaseSettings):
    """Add configuration to default Pydantic BaseSettings."""

    class Config:
        """Configure BaseSettings behavior."""

        extra = "forbid"
        use_enum_values = True
        env_prefix = "jv_"
class MatformerConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.cgcnn."""

    name: Literal["matformer"]
    conv_layers: int = 5
    edge_layers: int = 0
    atom_input_features: int = 92
    edge_features: int = 128
    triplet_input_features: int = 40
    node_features: int = 128
    fc_layers: int = 1
    fc_features: int = 128
    output_features: int = 1
    node_layer_head: int = 4
    edge_layer_head: int = 4
    nn_based: bool = False

    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    use_angle: bool = False
    angle_lattice: bool = False
    classification: bool = False
    pre_train: bool = False
    position_noise: float = None
    lattice_noise: float = None
    mask_ratio: float = None
    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


class Matformer(nn.Module):
    """att pyg implementation."""

    def __init__(self, config: MatformerConfig = MatformerConfig(name="matformer")):
        """Set up att modules."""
        super().__init__()
        self.classification = config.classification
        self.use_angle = config.use_angle
        self.pre_train = config.pre_train
        self.mask_ratio = config.mask_ratio is not None
        self.position_noise = config.position_noise is not None
        self.lattice_noise = config.lattice_noise is not None
        self.atom_embedding = nn.Linear(
            config.atom_input_features, config.node_features
        )
        self.rbf = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_features,
            ),
            nn.Linear(config.edge_features, config.node_features),
            nn.Softplus(),
            nn.Linear(config.node_features, config.node_features),
        )
        self.att_layers = nn.ModuleList(
            [
                MatformerConv(in_channels=config.node_features, out_channels=config.node_features, heads=config.node_layer_head, edge_dim=config.node_features)
                for _ in range(config.conv_layers)
            ]
        )
        
        if not self.pre_train:
            self.fc = nn.Sequential(
                nn.Linear(config.node_features, config.fc_features), nn.SiLU()
            )
            self.sigmoid = nn.Sigmoid()
            
            self.link = None
            
            self.link_name = config.link
            if config.link == "identity":
                self.link = lambda x: x
            elif config.link == "log":
                self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            if not self.zero_inflated:
                self.fc_out.bias.data = torch.tensor(
                    np.log(avg_gap), dtype=torch.float
                )
        elif config.link == "logit":
            self.link = torch.sigmoid

        if self.pre_train:
            pass
        elif self.classification:
            self.fc_out = nn.Linear(config.fc_features, 2)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc_out = nn.Linear(
                config.fc_features, config.output_features
            )
        if self.mask_ratio:
            self.mlm_pred = nn.Linear(
                config.node_features, 119
            )
            self.softmax_mlm = nn.LogSoftmax(dim=-1)
        if self.position_noise:
            self.position_mlp = nn.Linear(
                config.node_features, 3
            )
        if self.lattice_noise:
            self.lattice_mlp = nn.Linear(
                config.node_features, 9
            )

    def forward(self, data) -> torch.Tensor:
        data, ldata = data
        # initial node features: atom feature network...
        collect_dict = {}
        #collect_list = []
        node_features = self.atom_embedding(data.x)
        edge_feat = torch.norm(data.edge_attr, dim=1)
        
        edge_features = self.rbf(edge_feat)
        node_features = self.att_layers[0](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[1](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[2](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[3](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[4](node_features, data.edge_index, edge_features)

        if self.pre_train:
            if self.mask_ratio:
                atom_prob = self.softmax_mlm(self.mlm_pred(node_features))
                collect_dict["atoms"] = atom_prob
                #collect_list.append(atom_prob)

            if self.position_noise:
                position_pred = self.position_mlp(node_features)
                collect_dict["positions"] = position_pred
                #collect_list.append(position_pred)

            if self.lattice_noise:
                crystal_features = scatter(node_features, data.batch, dim=0, reduce="mean")
                lattice_pred = self.lattice_mlp(crystal_features)
                collect_dict["lattice"] = lattice_pred.view(-1, 3, 3)
                #collect_list.append(lattice_pred.view(-1, 3, 3))
            #return [atom_prob, position_pred, lattice_pred.view(-1, 3, 3)]
            return collect_dict
            #return collect_list
        # crystal-level readout
        else:
            features = scatter(node_features, data.batch, dim=0, reduce="mean")
            
            # features = F.softplus(features)
            features = self.fc(features)
            

            out = self.fc_out(features)

            if self.link:
                out = self.link(out)
            
            if self.classification:
                out = self.softmax(out)

            return torch.squeeze(out)


