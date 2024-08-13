# InterState 

Atoms re-arrange themselves during synthesis reactions, and understanding this choreography is critical to being able to predict synthesis recipes for novel materials. In this work, we show how equivariant neural networks can be used with minimal supervision to learn how to accurately identify the reaction paths that atoms take during changes in state. This method - called **InterState** - is showcased for model systems, showing the power of machine learning for accelerating materials design and discovery. 
This repository constains our framework implementation. In technical terms, we used E(3)-GNN to determine committor functions from atomistic simulations in order to create potentials that bias atomistic systems towards their transtion state. 

<p align="center">
  <img src="docs/figures/logo.gif" width="200" />
</p>

## Instalation 

```bash
# To install the latest git commit 
git clone https://github.com/TRI-AMDD/interstate 
cd interstate
pip install -e .
```

Alternary, connect to the EC2 instance named **Killian_GPU**, and do: 
```bash 
cd /home/ubuntu/ksheriff/PACKAGES/interstate
git pull
```
## Example of usage

The scripts and data necessary to reproduce the figures of the paper can be found in the [examples/](examples/) folder.

## Citing
If you use this repository in your work, please cite:

```
@article{}
```

## Contact
If any issues, feel free to contact me at:
```
Killian Sheriff
ksheriff at mit dot edu
```

## Thank you
Big thanks to Steven Torrisi and Amalie Trewartha for mentoring me over those 3 months at TRI! It was super fun working all together! :) 
