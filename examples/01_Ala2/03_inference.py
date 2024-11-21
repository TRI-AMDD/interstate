import os

import numpy as np
import torch
from interstate.dataset import GromacsDataset
from interstate.models import Committor
from mlcolvar.utils.io import create_dataset_from_files
from torch_geometric.data.lightning import LightningDataset

torch.set_default_dtype(torch.float64)


ROOT = "data/inference/"
os.makedirs(ROOT, exist_ok=True)

# Load best model
MODEL = Committor.load_from_checkpoint("data/training/ckpt/best.ckpt")
MODEL.eval()

"""
Evaluate our guessed committor + discovered CV on Parinello biased Ala2 trajectories 
"""


def get_parinello_data():

    root_dump = "data/parinello_paper/Committor/ala2/traj/"
    filenames = [
        root_dump + "alanine.A.0.xtc",
        root_dump + "alanine.B.0.xtc",
        root_dump + "alanine.A.1.xtc",
        root_dump + "alanine.B.1.xtc",
        root_dump + "alanine.A.2.xtc",
        root_dump + "alanine.B.2.xtc",
        root_dump + "alanine.A.3.xtc",
        root_dump + "alanine.B.3.xtc",
        root_dump + "alanine.A.4.xtc",
        root_dump + "alanine.B.4.xtc",
        root_dump + "alanine.A.5.xtc",
        root_dump + "alanine.B.5.xtc",
    ]

    os.system(f"rm -rf {ROOT}/dataset")
    group = [0, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    dataset = GromacsDataset(
        root=ROOT + "dataset",
        filenames=filenames,
        load_args="data/parinello_paper/Committor/ala2/data/input.ala2.pdb",
        group=group,
    )

    datamodule = LightningDataset(
        train_dataset=None,
        val_dataset=dataset,
        batch_size=100,
        num_workers=3,
        # shuffle=False,
    )

    return datamodule


def main_inference_on_parinello_data():

    datamodule = get_parinello_data()

    committor = np.zeros(len(datamodule.val_dataset))
    discovered_cv = np.zeros((len(datamodule.val_dataset), 2))

    with torch.no_grad():
        i = 0
        for x in datamodule.val_dataloader():  # this has shuffle=False

            graph = x.to("cuda")

            # cvs = model.e3gnn(graph).cpu().numpy()

            preds, cvs = MODEL(graph)
            preds = preds.cpu().numpy()
            cvs = cvs.cpu().numpy()

            for j in range(len(preds)):
                committor[i] = preds[j]
                discovered_cv[i] = cvs[j]
                i += 1

    np.save(ROOT + "guessed_committor_of_parinello_biased_trajectory.npy", committor)
    np.save(ROOT + "discovered_cv_of_parinello_biased_trajectory.npy", discovered_cv)


"""
Sampling of our discovered collective variable space to find the TSE.                 
Use previous results to estimate a good range that focuson the TSE. 
"""


def main_sampling_of_discovered_cv_space(cv1_range, cv2_range, npoints):

    cv1, cv2 = np.meshgrid(
        np.linspace(cv1_range[0], cv1_range[1], npoints),
        np.linspace(cv2_range[0], cv2_range[1], npoints),
    )
    q = np.zeros_like(cv1)
    for i in range(npoints):
        for j in range(npoints):
            cvs = torch.Tensor([cv1[i, j], cv2[i, j]]).to("cuda")

            # Only perform feed forward neural network part
            q[i, j] = MODEL.sigmoid(MODEL.nn(cvs)).detach().cpu().numpy()

    np.save(ROOT + "sampling_cv1.npy", cv1)
    np.save(ROOT + "sampling_cv2.npy", cv2)
    np.save(ROOT + "sampling_committor.npy", q)


if __name__ == "__main__":
    main_inference_on_parinello_data()
    main_sampling_of_discovered_cv_space(
        cv1_range=[-5, 15],
        cv2_range=[-15, 5],
        npoints=50,
    )
