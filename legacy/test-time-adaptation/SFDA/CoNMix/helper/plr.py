import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

def plr(prev_pseudo_labels, pseudo_labels, soft_output, class_num:int,  alpha=0.9) -> np.ndarray:
    pass