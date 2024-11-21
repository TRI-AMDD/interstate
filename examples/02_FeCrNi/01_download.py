import os

import gdown

url = (
    "https://drive.google.com/file/d/1t0OZGUmhsy3GjizJQOB30gb9eoWtjD4O/view?usp=sharing"
)
out_file = "01_unbiased_md.zip"

gdown.download(url, out_file, quiet=False)
os.system(f"unzip {out_file}")
