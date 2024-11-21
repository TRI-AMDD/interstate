import os

import lightning
import torch
from interstate.dataset import GromacsDataset
from interstate.models import Committor
from interstate.utils import initialize_committor_masses
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from mlcolvar.utils.trainer import MetricsCallback
from torch_geometric.data.lightning import LightningDataset

# Set seed for reproducibility
seed_everything(42)
torch.set_default_dtype(torch.float64)

ROOT = "data/training/"
USE_WANDB = True
BATCH_SIZE = 1200


def main():

    dim_out_feat = 2
    one_hot_dim = 4
    frac_train_val = 0.8

    atomic_masses = initialize_committor_masses(
        atom_types=[0, 1, 2, 3], masses=[1, 1, 1, 1], n_dims=3
    )  # [[[1,1,1]]*22][0]

    model = Committor(
        layers=[dim_out_feat, 32, 32, 1],
        mass=atomic_masses,
        alpha=1,  # 1e-1,
        delta_f=0.1,
        batch_size=BATCH_SIZE,
        options={
            "e3gnn": {
                "irreps_node_input": f"3x0e",  # coordinares, or cte
                "irreps_node_attr": f"{one_hot_dim}x0e",  # 1 hot atomic types, assume only 2 for now
                "irreps_edge_attr": "1x0e",
                "irreps_node_output": f"{dim_out_feat}x0e",
                "max_radius": 6,  # 5
                "num_neighbors": 10,  # 1
                "num_nodes": 10,  # 1
                "min_radius": 0.0,
                "mul": 25,
                "layers": 1,
                "lmax": 2,
                "pool_nodes": True,
                "number_of_basis": 10,
                "fc_neurons": 100,
                "pool_nodes": True,
            },
            "optimizer": {"lr": 1e-3, "weight_decay": 1e-5},
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ExponentialLR,
                "gamma": 0.99999,
            },
        },
    )

    # create dataset
    os.system(f"rm -rf {ROOT}dataset")

    root_dump = "data/parinello_paper/Committor/ala2/traj/"
    filenames = [
        root_dump + "alanine.A.0.xtc",
        root_dump + "alanine.B.0.xtc",
    ]

    group = [0, 1]
    dataset = GromacsDataset(
        root=ROOT + "dataset",
        filenames=filenames,
        load_args="data/parinello_paper/Committor/ala2/data/input.ala2.pdb",
        group=group,
    )
    dataset.shuffle()

    train_dataset = dataset[: int(len(dataset) * frac_train_val)]
    val_dataset = dataset[int(len(dataset) * frac_train_val) :]

    datamodule = LightningDataset(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=3,
        pin_memory=True,
    )

    # define callbacks
    metrics = MetricsCallback()
    logger = (
        WandbLogger(log_model="all", name="Committor Ala2", save_dir=ROOT)
        if USE_WANDB
        else None
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=ROOT + "ckpt/",
        # filename="{epoch}-{val_loss:.5f}",
        filename="best",
        monitor="valid_loss",
        save_top_k=1,
    )

    early_stopping = EarlyStopping(monitor="valid_loss", mode="min")
    # initialize trainer
    trainer = lightning.Trainer(
        default_root_dir=ROOT,
        callbacks=[
            metrics,
            checkpoint_callback,
            early_stopping,
        ],
        max_epochs=5000,
        logger=logger,
        enable_checkpointing=True,
        check_val_every_n_epoch=5,
        log_every_n_steps=5,
    )

    # fit model
    trainer.fit(
        model,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
    )

    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         print(f"Parameter: {name}, Gradient norm: {param.grad.norm()}")


if __name__ == "__main__":
    main()
