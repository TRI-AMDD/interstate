#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Committor function Loss Function. Modified from https://github.com/luigibonati/mlcolvar/. 
"""

__all__ = ["CommittorLoss", "committor_loss"]

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================
from typing import Dict, List, Optional, Tuple, Union

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================
import torch
import torch_geometric
from torch_geometric.data import Data


@torch.jit.script
def gradient(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y)]
    grad = torch.autograd.grad(
        [
            y,
        ],
        [x],
        grad_outputs=grad_outputs,
        create_graph=True,
    )

    # optional type refinement using an if statement
    if grad is None:
        grad = torch.zeros_like(x)
    else:
        grad = grad[0]

    # optional type refinement using an assert
    assert grad is not None
    return grad  # Now grad is always a torch.Tensor instead of Optional[torch.Tensor]


class CommittorLoss(torch.nn.Module):
    """Compute a loss function based on Kolmogorov's variational principle for the determination of the committor function"""

    def __init__(
        self,
        mass: torch.Tensor,
        alpha: float,
        cell: Optional[float] = 1.0,
        gamma: float = 10000,
        epsilon: float = 1.0,
        delta_f: float = 0,
    ):
        """Compute Kolmogorov's variational principle loss and impose boundary conditions on the metastable states

        Parameters
        ----------
        mass : torch.Tensor
            Atomic masses of the atoms in the system
        alpha : float
            Hyperparamer that scales the boundary conditions contribution to loss, i.e. alpha*(loss_bound_A + loss_bound_B)
        cell : float, optional
            CUBIC cell size length, used to scale the positions from reduce coordinates to real coordinates, by default None
        gamma : float, optional
            Hyperparamer that scales the whole loss to avoid too small numbers, i.e. gamma*(loss_var + loss_bound), by default 10000
        delta_f : float, optional
            Delta free energy between A (label 0) and B (label 1), units is kBT, by default 0.
            State B is supposed to be higher in energy.
        """
        super().__init__()
        self.mass = mass
        self.alpha = alpha
        self.cell = cell
        self.gamma = gamma
        self.delta_f = delta_f
        self.epsilon = epsilon

    def forward(
        self,
        x: torch.Tensor,  # Union[Data, Dict[str, torch.Tensor]],
        q: torch.Tensor,
        labels: torch.Tensor,
        w: torch.Tensor,
        cv: torch.Tensor,
        create_graph: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return committor_loss(
            x=x,
            q=q,
            labels=labels,
            w=w,
            mass=self.mass,
            alpha=self.alpha,
            gamma=self.gamma,
            epsilon=self.epsilon,
            delta_f=self.delta_f,
            create_graph=create_graph,
            cell=self.cell,
            cv=cv,
        )


def committor_loss(
    x: torch.Tensor,  # Union[Data, Dict[str, torch.Tensor]],
    q: torch.Tensor,
    labels: torch.Tensor,
    w: torch.Tensor,
    mass: torch.Tensor,
    cv: torch.Tensor,
    alpha: float,
    gamma: float = 10000,
    epsilon: float = 1.0,
    delta_f: float = 0,
    create_graph: bool = True,
    cell: Optional[float] = 1.0,
):
    """Compute variational loss for committor optimization with boundary conditions

    Parameters
    ----------
    x : torch.Tensor
        Input of the NN
    q : torch.Tensor
        Committor quess q(x), it is the output of NN
    labels : torch.Tensor
        Labels for states, A and B states for boundary conditions
    w : torch.Tensor
        Reweighing factors to Boltzmann distribution. This should depend on the simulation in which the data were collected.
    mass : torch.Tensor
        List of masses of all the atoms we are using, for each atom we need to repeat three times for x,y,z.
        Can be created using `committor.utils.initialize_committor_masses`
    alpha : float
        Hyperparamer that scales the boundary conditions contribution to loss, i.e. alpha*(loss_bound_A + loss_bound_B)
    gamma : float
        Hyperparamer that scales the whole loss to avoid too small numbers, i.e. gamma*(loss_var + loss_bound)
        By default 10000
    delta_f : float
        Delta free energy between A (label 0) and B (label 1), units is kBT, by default 0.
    create_graph : bool
        Make loss backwardable, deactivate for validation to save memory, default True
    cell : float
        CUBIC cell size length, used to scale the positions from reduce coordinates to real coordinates, default None

    Returns
    -------
    loss : torch.Tensor
        Loss value.
    gamma*loss_var : torch.Tensor
        The variational loss term
    gamma*alpha*loss_A : torch.Tensor
        The boundary loss term on basin A
    gamma*alpha*loss_B : torch.Tensor
        The boundary loss term on basin B
    """
    # inherit right device

    device = x.device  # x["pos"].device  #!!!

    mass = mass.to(device)

    # Create masks to access different states data
    # breakpoint()
    # mask_A = torch.nonzero(labels.squeeze() == 0, as_tuple=True)
    # mask_B = torch.nonzero(labels.squeeze() == 1, as_tuple=True)
    mask_A = torch.nonzero(labels.squeeze() == 0).squeeze()
    mask_B = torch.nonzero(labels.squeeze() == 1).squeeze()

    # Update weights of basin B using the information on the delta_f
    delta_f = torch.tensor([delta_f])
    if delta_f < 0:  # B higher in energy --> A-B < 0
        w[mask_B] = w[mask_B] * torch.exp(delta_f.to(device))
    elif delta_f > 0:  # A higher in energy --> A-B > 0
        w[mask_A] = w[mask_A] * torch.exp(delta_f.to(device))

    ###### VARIATIONAL PRINICIPLE LOSS ######
    # Each loss contribution is scaled by the number of samples

    # We need the gradient of q(x)
    # grad_outputs = torch.ones_like(q)
    # grad = torch.autograd.grad(
    #     q,
    #     x["pos"],
    #     grad_outputs=grad_outputs,
    #     retain_graph=True,
    #     create_graph=create_graph,
    # )[
    #     0
    # ]  #!!!
    # Reshaping gradiant to fit batch size
    # grad = grad.reshape((len(x.labels), len(x.pos) // len(x.labels), 2))  # !!!

    #!!! TRY DIFF WRT TO E3GNN output

    grad = gradient(q, cv)
    # grad_outputs = [torch.ones_like(q),]

    # grad = torch.autograd.grad(
    #     [q,],
    #     [cv,],
    #     grad_outputs=grad_outputs,
    #     retain_graph=True,
    #     create_graph=create_graph,
    # )[
    #     0
    # ]  #!!!
    # grad_old = grad = torch.autograd.grad(
    #     q,
    #     cv,
    #     grad_outputs=grad_outputs,
    #     retain_graph=True,
    #     create_graph=create_graph,
    # )[
    #     0
    # ]  #!!!

    # # # #Reshaping gradiant to fit batch size
    mass = mass[
        : len(cv[0])
    ]  #!!! NEED TO FIX AND THNK OF THIS, not sure I underatndh hoe nass work

    # from torchviz import make_dot
    # dot = make_dot(q, params={'x.pos': x.pos})
    # dot.render(filename='computation_graph', format='png')

    # TODO this fixes cell size issue
    if cell is not None:
        grad = grad / cell

    # we sanitize the shapes of mass and weights tensors
    # mass should have size [1, n_atoms*spatial_dims]

    mass = mass.unsqueeze(0)
    # weights should have size [n_batch, 1]

    # mass = mass[0][: len(grad[0])] #!!!

    w = w.unsqueeze(-1)

    # we get the square of grad(q) and we multiply by the weight
    weighted_grad = torch.pow(grad, 2) * (1 / mass)

    grad_square = (
        torch.sum(weighted_grad, dim=1, keepdim=True) * w
    )  #!!! keep dim used to be true

    # variational contribution to loss: we sum over the batch

    loss_var = torch.mean(grad_square)

    # print(loss_var)

    # boundary conditions

    q_A = q[mask_A]
    q_B = q[mask_B]

    # assert (len(q_A)!=0) or (len(q_B)!=0)
    # breakpoint()

    # !!! if the batch has no label in B then loss is nan which mess up things
    # loss_A = torch.mean(torch.pow(q_A, 2))
    # loss_B = torch.mean(torch.pow((q_B - 1), 2))
    if len(q_A) > 0:
        loss_A = torch.mean(torch.pow(q_A, 2))
    else:
        loss_A = torch.tensor(0)

    if len(q_B) > 0:
        loss_B = torch.mean(torch.pow((q_B - 1), 2))
    else:
        loss_B = torch.tensor(0)

    loss = gamma * loss_var * epsilon + gamma * alpha * (loss_A + loss_B)

    # print(f"{loss_var=}")
    # print(f"{q_A.mean()}")
    # print(f"{q_B.mean()}")

    # TODO maybe there is no need to detach them for logging
    return (
        loss,
        gamma * loss_var.detach(),
        alpha * gamma * loss_A.detach(),
        alpha * gamma * loss_B.detach(),
    )
