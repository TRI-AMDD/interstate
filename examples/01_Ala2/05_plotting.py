import os

import lovelyplots
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.style.use("paper")
import numpy as np
import matplotlib

os.makedirs("figures", exist_ok=True)

# Fessa colormap
paletteFessa = [
    "#1F3B73",  # dark-blue
    "#2F9294",  # green-blue
    "#50B28D",  # green
    "#A7D655",  # pisello
    "#FFE03E",  # yellow
    "#FFA955",  # orange
    "#D6573B",  # red
]
CM_FESSA = LinearSegmentedColormap.from_list("fessa", paletteFessa)


"""
Evaluate our guessed committor + discovered CV on Parinello biased Ala2 trajectories 
"""

ROOT = "data/dihedrals/"
phi = np.load(ROOT + "phi_of_parinello_biased_trajector.npy")
theta = np.load(ROOT + "theta_of_parinello_biased_trajector.npy")
phi, theta = np.deg2rad(phi), np.deg2rad(theta)

ROOT = "data/inference/"
committor = np.load(ROOT + "guessed_committor_of_parinello_biased_trajectory.npy")
discovered_cv = np.load(ROOT + "discovered_cv_of_parinello_biased_trajectory.npy")

cv1 = np.load(ROOT + "sampling_cv1.npy")
cv2 = np.load(ROOT + "sampling_cv2.npy")
q = np.load(ROOT + "sampling_committor.npy")

fig, ax = plt.subplots()

ax.set_ylim((-1.8, 1))
ax.set_xlim((-4, 2))
ax.set_xlabel(r"$\phi$",fontsize=10,)
ax.set_ylabel(r"$\theta$",fontsize=10,)

# Superimpose Parinello's dihedral angles and our guessed committor.

# Reducing transparency of atoms not considered in the TSE (committor close to 0.5)

try:

    alphas = np.ones(len(committor))
    mask = (committor < 0.45) | (committor > 0.55)

    # Systems not in committor plotted with transparency of 0.1
    transparency = 1
    ax.scatter(
        phi[mask],
        theta[mask],
        c=committor[mask],
        vmin=0,
        vmax=1,
        cmap=CM_FESSA,
        alpha=np.ones(len(phi[mask])) * transparency,
        zorder=1,
    )

    # Systems in committor plotted with transparency of 1, and high zorder
    transparency = 1
    sc = ax.scatter(
        phi[~mask],
        theta[~mask],
        c=committor[~mask],
        vmin=0,
        vmax=1,
        cmap=CM_FESSA,
        alpha=np.ones(len(phi[~mask])) * transparency,
        zorder=99,
        label=r"E(3)-Committor (our work)"
    )
except:
    print("Error, plotting all data...")
    transparency = 0.5
    sc = ax.scatter(
        phi,
        theta,
        c=committor,
        vmin=0,
        vmax=1,
        cmap=CM_FESSA,
        alpha=transparency,
        # zorder=99,
    )

phi_pnas, theta_pnas = np.loadtxt('data/pnas.txt', unpack=True)
phi_pnas, theta_pnas = np.deg2rad(phi_pnas), np.deg2rad(theta_pnas)
ax.plot(phi_pnas, theta_pnas, 's',color='#b08968',label='TS of ref. 35', alpha=transparency, zorder=999,markersize=0.95)

cbar = plt.colorbar(
    sc,
)
cbar.set_label(r"Learned committor $q$", fontsize=10, labelpad=15, rotation=270)
ax.legend(loc="best", frameon=True)
fig.savefig("figures/Ala2_dihedrals_with_transparency.png", transparent=False)


"""
Sampling of our discovered collective variable space to find the TSE.                 
Use previous results to estimate a good range that focuson the TSE. 
"""

# Plot contours
fig, ax = plt.subplots()
cs = ax.contourf(cv1, cv2, q, 10, cmap=CM_FESSA)
ax.set_xlabel(r"Learned reaction coordinate $r_1$",fontsize=10)
ax.set_ylabel(r"Learned reaction coordinate $r_2$",fontsize=10)
cbar = fig.colorbar(cs)
cbar.ax.set_ylabel(r"Learned committor $q$",fontsize=10,rotation=270,labelpad=15 )
fig.savefig("figures/sampling_discovered_cvs.pdf", transparent=False)

