from MagneticField import calculate_field
from MagneticForce import calculate_force
import numpy as np


def construct_matrix(poses, moment, target_moment, t):
    field_values = []
    force_values = []
    for pos, rotation in poses:
        field_values.append(calculate_field(moment, t, pos, rotation))
        force_values.append(calculate_force(moment, pos, rotation, t, target_moment))

    return np.concatenate((field_values, force_values), axis=1)