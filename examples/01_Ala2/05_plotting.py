import os

import lovelyplots
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.style.use("paper")
import numpy as np

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

# Add committor line (digitized) from figure 2b of Parinello paper
x = [
    -0.63996884,
    -0.5488306,
    -0.4109317,
    -0.30793908,
    -0.14524679,
    0.00936435,
    0.2126457,
    0.31278166,
    0.45344556,
    0.56898,
    0.58738791,
]
y = [
    1.2352724,
    0.97568781,
    0.69404684,
    0.50629637,
    0.21639093,
    -0.05418493,
    -0.4241635,
    -0.62020389,
    -0.90184232,
    -1.20560334,
    -1.29122175,
]

ax.plot(x, y, "-k", label="Committor [X]")
ax.set_ylim((-1, 1))
ax.set_xlim((-4, 2))
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$\theta$")
fig.savefig("figures/Ala2_dihedrals_line.png", transparent=False)

# Superimpose Parinello's dihedral angles and our guessed committor.

# Reducing transparency of atoms not considered in the TSE (committor close to 0.5)

try:
    alphas = np.ones(len(committor))
    mask = (committor < 0.49) | (committor > 0.51)

    # Systems not in committor plotted with transparency of 0.1
    transparency = 0.1
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
    )
except:
    print("Error, plotting all data...")
    transparency = 1
    sc = ax.scatter(
        phi,
        theta,
        c=committor,
        vmin=0,
        vmax=1,
        cmap=CM_FESSA,
        alpha=0.5,
        zorder=99,
    )


# Colorbar and legends
cbar = plt.colorbar(
    sc,
)
cbar.set_label("Guessed committor", rotation=270, labelpad=15)
ax.legend(loc="best", frameon=True)
fig.savefig("figures/Ala2_dihedrals_with_transparency.png", transparent=False)

# Plotting density maps (hexbin) for datapoints considered to be in TSE (committor close to 0.5)
fig, ax = plt.subplots()
hb = ax.hexbin(phi[~mask], theta[~mask], gridsize=50, cmap="inferno", mincnt=1)
# ax.plot(x, y, "r-", label="Committor [X]")
cb = fig.colorbar(hb, ax=ax, label="Counts")

ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$\theta$")
ax.set_ylim((-1, 1))
ax.set_xlim((-4, 2))
fig.savefig("figures/Ala2_hexbin.png", transparent=False)

# Plotting phi histigram for datapoints considered to be in TSE (committor close to 0.5)
fig, ax = plt.subplots()
ax.hist(phi[~mask], density=True, bins=200)

ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"Density")
# ax.set_xlim((-4, 2))
fig.savefig("figures/Ala2_phihist.png", transparent=False)

# Plotting our discovered collection variable evaluate on the Parinello snapshots
fig, ax = plt.subplots()
sc = ax.scatter(
    discovered_cv[:, 0],
    discovered_cv[:, 1],
    c=committor,
    vmin=0,
    vmax=1,
    cmap=CM_FESSA,
    # alpha=alphas,
)
cbar = plt.colorbar(sc)
cbar.set_label("Guessed committor", rotation=270, labelpad=15)
ax.set_xlabel(r"$CV_0$")
ax.set_ylabel(r"$CV_1$")
fig.savefig("figures/Ala2_discovered_CV.png", transparent=False)

# Plotting histogram of committor values in the TSE (committor close to 0.5)
fig, ax = plt.subplots()
ax.hist(committor[~mask], bins=200, density=True)
ax.set_xlabel(r"Guessed committor")
ax.set_ylabel(r"Density")
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/committor_histogram.png", transparent=False)


"""
Sampling of our discovered collective variable space to find the TSE.                 
Use previous results to estimate a good range that focuson the TSE. 
"""

# Plot contours
fig, ax = plt.subplots()
cs = ax.contourf(cv1, cv2, q, 10, cmap=CM_FESSA)

# Note that in the following, we explicitly pass in a subset of the contour
# levels used for the filled contours.  Alternatively, we could pass in
# additional levels to provide extra resolution, or leave out the *levels*
# keyword argument to use all of the original levels.

# cs2 = ax.contour(cs, levels=cs.levels[::2], colors="r")

ax.set_xlabel(r"$CV_1$")
ax.set_ylabel(r"$CV_2$")

# Make a colorbar for the ContourSet returned by the contourf call.
cbar = fig.colorbar(cs)
cbar.ax.set_ylabel("Guessed committor")
# cbar.add_lines(cs2)

fig.savefig("figures/sampling_discovered_cvs.png", transparent=False)

# TODO: Let's keep in mind to check how many datapoints are classified as state A or B, yet have theta and phi angle falling onto the TSE






