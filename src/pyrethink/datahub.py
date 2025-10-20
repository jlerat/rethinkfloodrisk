from pathlib import Path
import re
import json
import numpy as np
import pandas as pd

FHERE = Path(__file__).resolve().parent
FROOT = FHERE.parent.parent

ENV = "LOCAL"

with (FHERE / "config.json").open("r") as fo:
    CONFIG = json.load(fo)[ENV]

def replace_root(path):
    root_label = "package_root_folder"
    if path.startswith(root_label):
        return FROOT / re.sub(root_label + "/", "", path)
    else:
        return path

DATA_FOLDER = replace_root(CONFIG["data_folder"])



