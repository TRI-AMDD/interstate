import os

import numpy as np
import torch
from interstate.dataset import OvitoDataset
from ovito.io import export_file, import_file
from torch_geometric.data.lightning import LightningDataset

torch.set_default_dtype(torch.float64)


def per_atom_inference(root, model, filenames):

    os.makedirs(root + "per_atom_committor/", exist_ok=True)
    model.to("cpu")
    model.eval()

    # Removing pooling
    model.e3gnn.pool_nodes = False

    # os.system(f"rm -rf {root}dataset_per_snapshots")

    for filename in filenames:
        print(filename)
        dataset = OvitoDataset(
            root=root + "dataset_per_snapshots/" + os.path.basename(filename),
            filenames=[filename],
            load_args=None,
            group=[0, 1],
        )
        datamodule = LightningDataset(
            train_dataset=None,
            val_dataset=dataset,
            batch_size=1,
            num_workers=3,
            shuffle=False,
        )
        with torch.no_grad():
            i = 0
            for x in datamodule.val_dataloader():

                graph = x.to("cpu")

                # cvs = model.e3gnn(graph).cpu().numpy()

                preds, cvs = model(graph)
                preds = preds.cpu().numpy()
                cvs = cvs.cpu().numpy()

            pipeline = import_file(filename)
            data = pipeline.compute(0)
            comm = preds.squeeze()
            data.particles_.create_property("committor", data=comm)

            export_file(
                data,
                root + "per_atom_committor/" + os.path.basename(filename),
                "lammps/dump",
                columns=[
                    "Particle Identifier",
                    "Particle Type",
                    "Position.X",
                    "Position.Y",
                    "Position.Z",
                    "committor",
                ],
            )


def global_inference(root, model, filenames):
    model.to("cpu")
    model.eval()

    # Removing pooling
    model.e3gnn.pool_nodes = True

    # os.system(f"rm -rf {root}global_dataset")
    dataset = OvitoDataset(
        root=root + "global_dataset", filenames=filenames, load_args=None, group=[0, 1]
    )
    datamodule = LightningDataset(
        train_dataset=None,
        val_dataset=dataset,
        batch_size=1,
        num_workers=3,
        shuffle=False,
    )

    committor = np.zeros(len(filenames))
    discovered_cv = np.zeros((len(filenames), 2))
    with torch.no_grad():
        i = 0
        for x in datamodule.val_dataloader():
            print(i)

            # graph = x.to("cuda")
            graph = x

            # cvs = model.e3gnn(graph).cpu().numpy()
            preds, cvs = model(graph)
            preds = preds.cpu().numpy()
            cvs = cvs.cpu().numpy()
            print(preds)

            for j in range(len(preds)):

                committor[i] = preds[j]
                discovered_cv[i] = cvs[j]
                i += 1
    return committor, discovered_cv
