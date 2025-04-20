from ConstantParams import MU_0
import numpy as np 


def calculate_force(moment, p, angle, t, target_moment):
    r = t - p 
    r_norm = r / np.linalg.norm(r)
    r_hat = r / r_norm 

    m_hat = np.array([0, 0, moment])
    m_hat = m_hat @ angle 

    return 3 * MU_0 * moment / (4 * np.pi * r_norm ** 4) \
           * np.dot(
           np.outer(m_hat, r_hat) + 
           np.outer(r_hat, m_hat) - 
           ((5 * np.outer(r_hat, r_hat) - np.eye(3)) * np.dot(m_hat, r_hat)),
           target_moment)
