import os
import sys

import lightning
import torch

# JIT and torchscript
from e3nn.util.jit import script
from interstate.dataset import OvitoDataset
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

use_journey_loss = sys.argv[1]
ROOT = "data/committor/" if use_journey_loss == "True" else "data/classifier/"

BATCH_SIZE = 10


def main():

    dim_out_feat = 2
    one_hot_dim = 3
    frac_train_val = 0.8
    USE_WANDB = True

    atomic_masses = initialize_committor_masses(
        atom_types=[0,1,2], masses=[1,1,1], n_dims=3
    )  # [[[1,1,1]]*22][0]

    model = Committor(
        layers=[dim_out_feat, 32, 32, 1],
        mass=atomic_masses,
        alpha=1e-1,  # 1e-1, 1.0
        delta_f=0.1,
        batch_size=BATCH_SIZE,
        epsilon=(
            1.0 if use_journey_loss == "True" else 0.0
        ),  # set to 0 if only want classifier loss
        options={
            "e3gnn": {
                "irreps_node_input": f"3x0e",  # coordinares, or cte
                "irreps_node_attr": f"{one_hot_dim}x0e",  # 1 hot atomic types, assume only 2 for now
                "irreps_edge_attr": "1x0e",
                "irreps_node_output": f"{dim_out_feat}x0e",
                "max_radius": 5.0,  # 5
                "num_neighbors": 16,  # 1
                "num_nodes": 16,  # 1
                "min_radius": 0.0,
                "mul": 25,
                "layers": 1,
                "lmax": 2,
                "number_of_basis": 10,
                "fc_neurons": 100,
                "pool_nodes": True,  #!!! Pooling operation kills gradiants?
                "normalize": True,
            },
            "optimizer": {"lr": 1e-3, "weight_decay": 1e-5},
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ExponentialLR,
                "gamma": 0.99999,
            },
        },
    )

    # model = torch.compile(model)

    # create dataset
    os.system(f"rm -rf {ROOT}")

    group = [0, 1]  # [0, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    dataset = OvitoDataset(
        root=ROOT + "dataset",
        filenames=[
            "../../01_unbiased_md/04_start_and_end_states/data/stateA/snapshot/snapshots_800K.dump",
            "../../01_unbiased_md/04_start_and_end_states/data/stateB/snapshot/snapshots_300K.dump",
        ],
        load_args=None,
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
        shuffle=False,
        pin_memory=True,
    )

    # define callbacks
    metrics = MetricsCallback()
    logger = (
        WandbLogger(log_model="all", name="Committor FeCrNi", save_dir=ROOT)
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

    early_stopping = EarlyStopping(monitor="valid_loss", mode="min",patience=20)
    # initialize trainer
    trainer = lightning.Trainer(
        default_root_dir=ROOT,
        callbacks=[
            metrics,
            checkpoint_callback,
            early_stopping,
        ],
        max_epochs=250,
        logger=logger,
        enable_checkpointing=True,
        check_val_every_n_epoch=5,
        log_every_n_steps=2,
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

    # model = Committor.load_from_checkpoint("data/ckpt/best.ckpt")
    # mod_script = script(model)
    # torch.jit.save(mod_script, "data/ckpt/traced_best.pt")


if __name__ == "__main__":

    main()
