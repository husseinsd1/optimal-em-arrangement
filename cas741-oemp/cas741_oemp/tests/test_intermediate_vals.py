from cas741_oemp.MagneticMoment import calculate_moment
from cas741_oemp.InputParameters import load_params, Params
from cas741_oemp.ActuationMatrix import construct_matrix
from cas741_oemp.OptimalPlacement import find_opt_positions
from cas741_oemp.OutputResults import output_results
from cas741_oemp.GeneratePoses import generate_poses

import numpy as np
import json
import os
import pytest


HERE = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(HERE, 'config/test-intermediate-val.json')
with open(path, 'r') as f:
    try:
        data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError("Failed to decode JSON file. Please ensure it is valid JSON.") from e


@pytest.mark.parametrize("test_id,test_params", data.items())
def test_lowest_eig(test_id, test_params):
    config = test_params
    expected_eig = config["expected"]

    params = Params(
            config["N"], config["I"], config["A"],
            config["M"], config["K"], config["V"],
            np.array(config["t"],   dtype=float),
            np.array(config["m_t"], dtype=float),
            config["l"]
    )    
    
    moment = calculate_moment(params.N, params.I, params.A)
    poses = generate_poses(params.M, params.l)
    U_matrix = construct_matrix(poses, moment, params.m_t, params.t)
    _, lowest_eig = find_opt_positions(params.M, params.K, U_matrix, poses, params.l)

    assert pytest.approx(expected_eig, rel=1e-3) == float(lowest_eig), f"Expected lowest eigenval to be {expected_eig}, got {lowest_eig}"

