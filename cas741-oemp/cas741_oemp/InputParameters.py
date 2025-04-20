import numpy as np 
import json
import os

from ConstantParams import MAX_CURR


class Params:
    def __init__(self, N, I, A, M, K, V, t, m_t, l):
        self.N, self.I, self.A, self.M, self.K, self.V, self.t, self.m_t, self.l = N, I, A, M, K, V, t, m_t, l


def load_params():
    while True:
        path = input("Enter the path to the JSON config file: ")
        try:
            params = extract_params(path)
            return params 
        except Exception as e:
            print(f"Error: {e} \n")


def verify_path(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"The file at path {path} could not be found.")
    return True


def validate_params(data):
    """
        Constraints:
            N > 0
            A > 0
            0 < I <= MAX_CURR (20000)
            M > 0
            K > 0
            V > 0
            t is a real 3D numpy vector with t_z > 0
            m_t is a real 3D numpy vector
            0 < l <= cbrt(V) 
    """

    if "N" not in data:
        raise ValueError("Missing 'N' in config file.")
    N = data["N"]
    if not (isinstance(N, int) and N > 0):
        raise ValueError("N must be a positive integer.")

    if "A" not in data:
        raise ValueError("Missing 'A' in config file.")
    A = data["A"]
    if not (isinstance(A, (int, float)) and A > 0):
        raise ValueError("A must be a positive number.")

    if "I" not in data:
        raise ValueError("Missing 'I' in config file.")
    I = data["I"]
    if not (isinstance(I, (int, float)) and I > 0 and I <= MAX_CURR):
        raise ValueError(f"I must be > 0 and <= {MAX_CURR}.")

    if "M" not in data:
        raise ValueError("Missing 'M' in config file.")
    M = data["M"]
    if not (isinstance(M, int) and M > 0):
        raise ValueError("M must be a positive integer.")
    
    if "K" not in data:
        raise ValueError("Missing 'K' in config file.")
    K = data["K"]
    if not (isinstance(K, int) and K > 0 and K <= M):
        raise ValueError("K must be an integer between 1 and M (inclusive).")
    
    if "V" not in data:
        raise ValueError("Missing 'V' in config file.")
    V = data["V"]
    if not (isinstance(V, (int, float)) and V > 0):
        raise ValueError("V must be a positive number.")

    if "t" not in data:
        raise ValueError("Missing 't' in config file.")
    t = data["t"]
    if not (isinstance(t, list) and len(t) == 3):
        raise ValueError("t must be a list of three numbers.")
    try:
        t_array = np.array(t, dtype=float)
    except Exception as e:
        raise ValueError("t must contain numeric values.")
    if t_array[2] <= 0:
        raise ValueError("The z component (third element) of t must be positive.")

    if "m_t" not in data:
        raise ValueError("Missing 'm_t' in config file.")
    m_t = data["m_t"]
    if not (isinstance(m_t, list) and len(m_t) == 3):
        raise ValueError("m_t must be a list of three numbers.")
    try:
        m_t_array = np.array(m_t, dtype=float)
    except Exception as e:
        raise ValueError("m_t must contain numeric values.")

    if "l" not in data:
        raise ValueError("Missing 'l' in config file.")
    l = data["l"]
    if not (isinstance(l, (int, float)) and l > 0 and l <= np.cbrt(V)):
        raise ValueError(f"l must be > 0 and <= the cube root of V ({np.cbrt(V)}).")

    return Params(N, I, A, M, K, V, t_array, m_t_array, l)


def extract_params(json_path):
    verify_path(json_path)

    with open(json_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError("Failed to decode JSON file. Please ensure it is valid JSON.") from e

    params = validate_params(data)
    return params