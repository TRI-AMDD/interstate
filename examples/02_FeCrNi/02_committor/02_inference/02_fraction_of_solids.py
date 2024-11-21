import os

import numpy as np
from ovito.io import import_file
from ovito.modifiers import PolyhedralTemplateMatchingModifier


def get_solid_fractions(root, filenames):
    # Initialize lists to store fractions for each type
    frac_of_solid_fcc = []
    frac_of_solid_hcp = []
    frac_of_solid = []

    # Process each file
    for file in filenames:
        pipeline = import_file(file)
        pipeline.modifiers.append(PolyhedralTemplateMatchingModifier())
        data = pipeline.compute()

        # Get structure types
        structure_types = data.particles.structure_types
        total_particles = len(structure_types)

        # Compute fractions
        frac_of_solid_fcc.append(np.sum(structure_types == 1) / total_particles)
        frac_of_solid_hcp.append(np.sum(structure_types == 2) / total_particles)
        frac_of_solid.append(np.sum(structure_types != 0) / total_particles)

    np.save(root + "frac_of_solid.npy", frac_of_solid)
    np.save(root + "frac_of_solid_fcc.npy", frac_of_solid_fcc)
    np.save(root + "frac_of_solid_hcp.npy", frac_of_solid_hcp)


if __name__ == "__main__":

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
    for i, data in enumerate(data_types):
        filenames = file_paths[i]
        root = f"data/solid_fractions/{data}/"
        os.makedirs(root, exist_ok=True)
        get_solid_fractions(root, filenames)
