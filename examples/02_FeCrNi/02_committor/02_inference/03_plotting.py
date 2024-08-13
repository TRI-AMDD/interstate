import lovelyplots
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
plt.style.use("paper")

datas = ["seed_growth","nucleation"]   
stepss = [np.arange(0, 500_000, 10_000) ,np.arange(0, 470_000, 10_000)]

# Plot committor 
for data, steps in zip(datas, stepss):
    classifier = np.load(f"data/classifier/{data}/committor.npy")
    committor = np.load(f"data/committor/{data}/committor.npy")

    ptm = np.load(f"data/solid_fractions/{data}/frac_of_solid.npy")

    fig, ax = plt.subplots()
    ax.plot(steps * 0.002 / 1000, ptm, ":d", label="PTM (heuristic classifier)", color="k")
    ax.plot(
        steps * 0.002 / 1000, classifier, "-s", label="$L= L_d$ (ML classifier)", color="k"
    )
    ax.plot(
        steps * 0.002 / 1000,
        committor,
        "-o",
        label="$L = L_d+L_j$ (our work)",
        color="#8a0329",
        zorder=999,
    )
    ax.fill_between(
        steps * 0.002 / 1000,
        0.45,
        0.55,
        alpha=0.3,
        color="#8a0329",
        label=r" TSE $q \in [0.45,0.55]$",
    )  #

    # ax.axhline(y=0.5, ls='--', color='k')

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel(r"Committor $q$")

    ax.legend(loc="best", frameon=True)
    fig.savefig(f"figures/{data}_classifier_and_committor_and_ptm_vs_time.pdf")


# Plot discovered CVs
 # Plot committor 
for data, steps in zip(datas, stepss):
    classifier_cv = np.load(f"data/classifier/{data}/discovered_cv.npy")
    committor_cv = np.load(f"data/committor/{data}/discovered_cv.npy")

    fig, ax = plt.subplots()
    sc = ax.scatter(classifier_cv[:,0], classifier_cv[:,1],c=steps*0.002/1000)
    ax.set_xlabel(r'$CV_0$')
    ax.set_ylabel(r'$CV_1$')
    cbar = plt.colorbar(sc,)
    cbar.ax.set_ylabel('Time (ns)', rotation=270, labelpad=20)
    fig.savefig(f'figures/{data}_dicovered_collective_variables_classifier.pdf')

    fig, ax = plt.subplots()
    sc = ax.scatter(committor_cv[:,0], committor_cv[:,1],c=steps*0.002/1000)
    ax.set_xlabel(r'$CV_0$')
    ax.set_ylabel(r'$CV_1$')
    cbar = plt.colorbar(sc,)
    cbar.ax.set_ylabel('Time (ns)', rotation=270, labelpad=20)
    fig.savefig(f'figures/{data}_dicovered_collective_variables_committor.pdf')

   