import os

import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis import dihedrals
from MDAnalysis.analysis.dihedrals import Dihedral
from MDAnalysis.core.groups import AtomGroup

ROOT = "data/dihedrals/"
os.makedirs(ROOT, exist_ok=True)
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


theta = []
phi = []
for filename in filenames:
    u = mda.Universe(
        "data/parinello_paper/Committor/ala2/data/input.ala2.pdb",
        filename,
        guess_bonds=True,
    )
    atoms = u.select_atoms("all")

    ag = AtomGroup([6 - 1, 5 - 1, 7 - 1, 9 - 1], u)
    results = Dihedral([ag]).run()
    theta.append(results.angles.flatten())

    ag = AtomGroup([5 - 1, 7 - 1, 9 - 1, 15 - 1], u)
    results = Dihedral([ag]).run()
    phi.append(results.angles.flatten())


theta = [item for sublist in theta for item in sublist]
theta = np.array(theta).squeeze()

phi = [item for sublist in phi for item in sublist]
phi = np.array(phi).squeeze()


np.save(ROOT + "phi_of_parinello_biased_trajector.npy", phi)
np.save(ROOT + "theta_of_parinello_biased_trajector.npy", theta)
