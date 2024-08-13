from typing import Dict, List, Tuple, Union

import lightning
import torch
from e3nn import o3
from interstate.cluster import periodic_radius_graph_v2
from mlcolvar.core import FeedForward
from mlcolvar.core.nn.utils import Custom_Sigmoid
from mlcolvar.cvs.committor import initialize_committor_masses
from torch_cluster import radius_graph
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

from .cvs import BaseCV
from .loss import CommittorLoss
from .message_passing import MessagePassing
from .radial_basis import soft_one_hot_linspace
from .scatter import scatter


class BaseCVGraph(BaseCV):
    """
    Base collective variable class.

    To inherit from this class, the class must define a BLOCKS class attribute.
    """

    def __init__(
        self,
        in_features,
        out_features,
        preprocessing: torch.nn.Module = None,
        postprocessing: torch.nn.Module = None,
        *args,
        **kwargs,
    ):
        """Base CV class options.

        Parameters
        ----------
        in_features : int
            Number of inputs of the CV model
        out_features : int
            Number of outputs of the CV model, should be the number of CVs
        preprocessing : torch.nn.Module, optional
            Preprocessing module, default None
        postprocessing : torch.nn.Module, optional
            Postprocessing module, default None

        """
        super().__init__(
            in_features, out_features, preprocessing, postprocessing, *args, **kwargs
        )

    @property
    def example_input_array(self):

        one_node_attr = list(
            torch.zeros(
                int(self.hparams["options"]["e3gnn"]["irreps_node_attr"].split("x")[0])
            )
        )
        one_node_attr[0] = 1

        return Data(
            pos=torch.tensor(
                [
                    list(
                        torch.ones(
                            int(
                                self.hparams["options"]["e3gnn"][
                                    "irreps_node_input"
                                ].split("x")[0]
                            )
                        ),
                    ),
                    list(
                        torch.ones(
                            int(
                                self.hparams["options"]["e3gnn"][
                                    "irreps_node_input"
                                ].split("x")[0]
                            )
                        )
                        * 2,
                    ),
                ],
                dtype=torch.get_default_dtype(),
            ),
            node_attr=torch.tensor(
                [one_node_attr, one_node_attr], dtype=torch.get_default_dtype()
            ),
            node_input=torch.ones(
                2,
                int(
                    self.hparams["options"]["e3gnn"]["irreps_node_input"].split("x")[0]
                ),
            ),
            batch=torch.zeros(2, dtype=torch.int64),
            labels=torch.tensor([0]),
            weights=torch.tensor([1]),
        )

    def validation_step(self, val_batch, batch_idx):
        """
        Equal to training step if not overridden. Different behaviors for train/valid step can be enforced in training_step() based on the self.training variable.
        """
        torch.set_grad_enabled(True)
        self.training_step(val_batch, batch_idx)


class E3GNN(lightning.LightningModule):
    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_node_output,
        max_radius,
        num_neighbors,
        num_nodes,
        min_radius=0.0,
        mul=50,
        layers=3,
        lmax=2,
        pool_nodes=True,
        number_of_basis=10,
        fc_neurons=100,
        normalize=False,
    ) -> None:
        super().__init__()

        self.lmax = lmax
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.number_of_basis = number_of_basis
        self.num_nodes = num_nodes
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.pool_nodes = pool_nodes
        self.fc_neurons = fc_neurons
        self.num_neighbors = num_neighbors
        self.normalize = normalize

        irreps_node_hidden = o3.Irreps(
            [(mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]]
        )

        self.mp = MessagePassing(
            irreps_node_input=irreps_node_input,
            irreps_node_hidden=irreps_node_hidden,
            irreps_node_output=irreps_node_output,
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr
            + o3.Irreps.spherical_harmonics(lmax),
            layers=layers,
            fc_neurons=[self.number_of_basis, self.fc_neurons],
            num_neighbors=num_neighbors,
        )

        self.irreps_node_input = self.mp.irreps_node_input
        self.irreps_node_attr = self.mp.irreps_node_attr
        self.irreps_node_output = self.mp.irreps_node_output

        self.sph = o3.SphericalHarmonics(
            list(range(self.lmax + 1)), normalize=True, normalization="component"
        )

    def forward_old(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:

        return data["pos"]

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        # x['pos'], x['batch'], x['node_attr'],x['node_input']
        # breakpoint()
        pos, batch, node_attr, node_input = x[0], x[1], x[2], x[3]

        # pos =
        # batch =
        # node_attr =
        # node_input=

        # edge_src, edge_dst = knn_graph(
        #     x=pos, k=self.num_neighbors, batch=batch, loop=True
        # )  # radius_graph(x=data.pos, r=self.max_radius, batch=data.batch, loop=True)
        edges = knn_graph(x=pos, k=self.num_neighbors, batch=batch, loop=True)
        edge_src = edges[0]
        edge_dst = edges[1]
        # breakpoint()

        # edge_index, vec = periodic_radius_graph_v2(x = pos[batch==0], r = 5, box = torch.tensor([10,10,10]).to('cuda') )

        edge_vec = pos[edge_src] - pos[edge_dst]

        if self.normalize:
            pass

        # !! Case where we only one one node which gives edge_vec = 0
        if torch.all(edge_vec == 0):
            edge_vec = pos

        edge_sh = self.sph(edge_vec)
        # edge_sh = o3.spherical_harmonics(
        #     range(self.lmax + 1), edge_vec, True, normalization="component"
        # )

        edge_attr = torch.ones(edge_sh.shape[0], 1).to(pos.device)  #!!!

        edge_attr = torch.cat([edge_attr, edge_sh], dim=1)

        # Edge length embedding
        edge_length = torch.norm(edge_vec, dim=1)
        # print(edge_length.max())
        # print(edge_length.min())
        # print('---')

        edge_length_embedding = soft_one_hot_linspace(
            edge_length,
            self.min_radius,
            self.max_radius,
            self.number_of_basis,
            basis="cosine",  # the cosine basis with cutoff = True goes to zero at max_radius
            cutoff=True,  # no need for an additional smooth cutoff
        ).mul(self.number_of_basis**0.5)

        # node_input = (
        #     data.pos
        # )  # torch.ones(data["node_attr"].shape[0], 1).to(data.pos.device)
        # # node_attr = data["node_attr"].to(device) #!!! data.pos for the MB potential
        # node_input = data["node_input"]#torch.ones(data["node_attr"].shape[0], 1).to(data.pos.device)

        node_outputs = self.mp(
            node_input,
            node_attr,
            edge_src,
            edge_dst,
            edge_attr,
            edge_length_embedding,
        )

        if self.pool_nodes:

            # return scatter(node_outputs, batch, dim=0).div(self.num_nodes**0.5)

            # self.num_nodes = len(batch[batch==0]) #!!!!
            # breakpoint()
            return scatter(node_outputs, batch, dim=0, reduce="mean").div(
                self.num_nodes**0.5
            )  #!!! Reduce used to be sum but that makes it depends on system size?
        else:

            return node_outputs


class Committor(BaseCVGraph, lightning.LightningModule):
    """Base class for data-driven learning of committor function.
    The committor function q is expressed as the output of a neural network optimized with a self-consistent
    approach based on the Kolmogorov's variational principle for the committor and on the imposition of its boundary conditions.
    TODO: Add reference upon publication

    **Data**: for training it requires a DictDataset with the keys 'data', 'labels' and 'weights'

    **Loss**: Minimize Kolmogorov's variational functional of q and impose boundary condition on the metastable states (CommittorLoss)

    References
    ----------
    .. [*] P. Kang, E. Trizio, and M. Parrinello, "Computing the Committor using the Committor: an Anatomy of the Transition state Ensemble", xxxx yy, 20zz

    See also
    --------
    mlcolvar.core.loss.CommittorLoss
        Kolmogorov's variational optimization of committor and imposition of boundary conditions
    mlcolvar.cvs.committor.utils.compute_committor_weights
        Utils to compute the appropriate weights for the training set
    mlcolvar.cvs.committor.utils.initialize_committor_masses
        Utils to initialize the masses tensor for the training
    """

    BLOCKS = ["e3gnn", "nn", "sigmoid"]

    def __init__(
        self,
        layers: list,
        mass: torch.Tensor,
        alpha: float,
        batch_size: int,
        gamma: float = 10000,
        delta_f: float = 0,
        epsilon: float = 1.0,
        cell: float = None,
        options: dict = None,
        **kwargs,
    ):
        """Define a NN-based committor model

        Parameters
        ----------
        layers : list
            Number of neurons per layer
        mass : torch.Tensor
            List of masses of all the atoms we are using, for each atom we need to repeat three times for x,y,z.
            The mlcolvar.cvs.committor.utils.initialize_committor_masses can be used to simplify this.
        alpha : float
            Hyperparamer that scales the boundary conditions contribution to loss, i.e. alpha*(loss_bound_A + loss_bound_B)
        gamma : float, optional
            Hyperparamer that scales the whole loss to avoid too small numbers, i.e. gamma*(loss_var + loss_bound), by default 10000
        delta_f : float, optional
            Delta free energy between A (label 0) and B (label 1), units is kBT, by default 0.
            State B is supposed to be higher in energy.
        cell : float, optional
            CUBIC cell size length, used to scale the positions from reduce coordinates to real coordinates, by default None
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default {}.
            Available blocks: ['nn'] .
        """
        super().__init__(in_features=layers[0], out_features=layers[-1], **kwargs)

        # =======  LOSS  =======
        self.loss_fn = CommittorLoss(
            mass=mass,
            alpha=alpha,
            gamma=gamma,
            delta_f=delta_f,
            cell=cell,
            epsilon=epsilon,
        )

        # ======= OPTIONS =======
        # parse and sanitize
        options = self.parse_options(options)

        # ======= BLOCKS =======
        # initialize NN turning
        o = "e3gnn"
        self.e3gnn = E3GNN(**options[o])  #!!!
        o = "nn"
        self.nn = FeedForward(layers, **options[o])

        # separately add sigmoid activation on last layer, this way it can be deactived
        o = "sigmoid"
        if (options[o] is not False) and (options[o] is not None):
            self.sigmoid = Custom_Sigmoid(**options[o])
        self.batch_size = batch_size

    def training_step(self, train_batch, batch_idx):
        """Compute and return the training loss and record metrics."""
        # =================get data===================

        x = train_batch  # !!!["data"]
        x["pos"].requires_grad = True  #!!!

        labels = train_batch["labels"]
        weights = train_batch["weights"]

        # =================forward====================
        # we use forward and not forward_cv to also apply the preprocessing (if present)

        q, cv = self.forward(x)

        # ===================loss=====================
        if self.training:
            loss, loss_var, loss_bound_A, loss_bound_B = self.loss_fn(
                x["pos"], q, labels, weights, cv=cv
            )
        else:

            loss, loss_var, loss_bound_A, loss_bound_B = self.loss_fn(
                x["pos"],
                q,
                labels,
                weights,
                cv=cv,
            )
        # ====================log=====================+
        name = "train" if self.training else "valid"

        self.log(
            f"{name}_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            f"{name}_loss_var", loss_var, on_epoch=True, batch_size=self.batch_size
        )
        self.log(
            f"{name}_loss_bound_A",
            loss_bound_A,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.log(
            f"{name}_loss_bound_B",
            loss_bound_B,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        return loss

    # def on_train_epoch_end(self):
    #     def plot_muller_brown():
    #         if self.current_epoch % 1 == 0:
    #             import os

    #             import matplotlib.pyplot as plt
    #             import numpy as np
    #             from plot import muller_brown_potential, plot_isolines_2D

    #             os.makedirs("figures/", exist_ok=True)

    #             fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    #             plot_isolines_2D(
    #                 self.forward,
    #                 ax=ax,
    #                 colorbar=True,
    #                 num_points=(25, 25),
    #                 given_forward=True,
    #                 vmin=0,
    #                 vmax=1,
    #             )
    #             plot_isolines_2D(
    #                 self.forward,
    #                 ax=ax,
    #                 colorbar=True,
    #                 levels=[0.5],
    #                 mode="contour",
    #                 linewidths=1,
    #                 num_points=(25, 25),
    #                 given_forward=True,
    #                 vmin=0,
    #                 vmax=1,
    #             )
    #             plot_isolines_2D(
    #                 muller_brown_potential,
    #                 levels=np.linspace(0, 24, 12),
    #                 ax=ax,
    #                 max_value=24,
    #                 colorbar=False,
    #                 mode="contour",
    #                 linewidths=1,
    #                 num_points=(25, 25),
    #                 vmin=0,
    #                 vmax=1,
    #             )
    #             plt.tight_layout()

    #             plt.savefig(f"figures/{self.current_epoch}.png")

    #     def plot_ala2():
    #         if self.current_epoch % 1 == 0:
    #             pass

    #     # plot_muller_brown()
    #     plot_ala2()
