# Downloading Ala2 atomistic simulations of state A, and state B, from https://www.nature.com/articles/s43588-024-00645-0#article-info

import os

os.makedirs("data/parinello_paper/", exist_ok=True)
os.system(
    "cd data/parinello_paper/ && git clone https://github.com/alphatestK/Committor"
)
