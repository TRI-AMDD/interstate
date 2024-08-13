import numpy as np
from interstate.inference import global_inference, per_atom_inference
from interstate.models import Committor

if __name__ == "__main__":

    model_types = ["committor", "classifier"]
    data_types = ["seed_growth", "nucleation"]
    file_paths = [
        [
            f"../../01_unbiased_md/02_seed_growth/02_growth/data/snapshot/snapshots_800K_{i}.dump"
            for i in range(0, 500_000, 10_000)
        ],
        [
            f"../../01_unbiased_md/03_nucleation/02_md/data/snapshot/snapshots_800K_{i}.dump"
            for i in range(0, 470_000, 10_000)
        ],
    ]

    for model_type in model_types:
        model = Committor.load_from_checkpoint(
            f"../01_training/data/{model_type}/ckpt/best.ckpt"
        )

        for i, data in enumerate(data_types):
            root = f"data/{model_type}/{data}/"

            per_atom_inference(root, model, file_paths[i])
            committor, discovered_cv = global_inference(root, model, file_paths[i])
            np.save(root + "committor.npy", committor)
            np.save(root + "discovered_cv.npy", discovered_cv)
