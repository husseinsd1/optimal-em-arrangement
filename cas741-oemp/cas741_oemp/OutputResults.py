import numpy as np


def output_results(x, K, poses):
    selected_indices = np.argsort(x)[-K:]
    selected_poses = [poses[i] for i in selected_indices]
    
    print("Selected indices:", selected_indices)
    print("Corresponding poses:")
    for idx, pose in zip(selected_indices, selected_poses):
        position, rotation = pose
        print(f"Index {idx}:")
        print(f"  Position: {position}")
        print(f"  Rotation matrix:\n{rotation}")