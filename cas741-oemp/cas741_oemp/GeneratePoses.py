import numpy as np
from scipy.spatial.transform import Rotation as R 


rng = np.random.default_rng(123)

def generate_poses(M, workspace_length):
    return [generate_pose(workspace_length) for i in range(M)]


def generate_pose(workspace_length):
    rotation = R.random(random_state=rng)
    pos = rng.random(3) * workspace_length
    return pos, rotation.as_matrix()