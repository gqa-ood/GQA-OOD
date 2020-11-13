import json
import numpy as np

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, file):
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

def entropy(v1):
    e = - (v1 * np.log(v1 + 1e-9)).sum()
    return e

def n_entropy(v1):
    e = entropy(v1)
    if len(v1) > 1:
        n = np.log(len(v1))
    else:
        n = e
    ne = entropy(v1)/n
    return ne

def std(v1, inv=True):
    std = np.std(v1)
    if inv:
        std = 1/(std + 1e-9)
    return std
