from ConstantParams import MU_0
import numpy as np 


def calculate_field(moment, t, p, angle):
    r = t - p 
    r_norm = r / np.linalg.norm(r)
    r_hat = r / r_norm 

    m_hat = np.array([0, 0, moment])
    m_hat = m_hat @ angle 

    return MU_0 * moment * (4 * np.pi * r_norm ** 3) * \
           ((3 * np.outer(r_hat, r_hat) - np.eye(3)) @ m_hat)
