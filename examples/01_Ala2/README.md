# Alanide dipeptide 

Application of **InterState** to atomistic simulations of alanide dipetitde. 

## Results 

#### How to find the TSE? 
You need to sample the discovered collective variable space! 

<p align="center">
  <img src="figures/sampling_discovered_cvs.png" width="400" />
</p>

#### Is our TSE physical? 
To awser this, we compare with the real transition state of alanide dipeptide characterized by $\theta \approx -\phi$ (black line).
The guessed committor is evaluated on biased trajectories of Ala2 (from Parinello) and compared with physical collective variables (dihedral angles). This suggest that framework correctly defines the transition state.

<p align="center">
  <img src="figures/Ala2_dihedrals_with_transparency.png" width="400" />
</p>

<p align="center">
  <img src="figures/Ala2_hexbin.png" width="400" />
</p>

#### Visualization of a Parinello biased simulations path in our discovered collective variable space 

<p align="center">
  <img src="figures/Ala2_discovered_CV.png" width="400" />
</p>

## Limitations 
1. Check number of points that are in the TSE region (correct $\theta$, $\phi$), yet don't have a committor value of $\approx 0.5$.
2. In GromacsDataset code, check that atom types are in the good order (see **TODO** block).