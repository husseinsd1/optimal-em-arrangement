import os
import json
import pytest
import numpy as np

from cas741_oemp.InputParameters import validate_params
from cas741_oemp.ConstantParams import MAX_CURR

JSON_PATH = os.path.join(os.path.dirname(__file__), "config/test-input-params.json")
with open(JSON_PATH, "r") as f:
    CASES = json.load(f)

EXPECTED = {
    "test-em-props-1":                    "N must be a positive integer.",
    "test-em-props-2":                    f"I must be > 0 and <= {MAX_CURR}.",
    "test-em-props-3":                    "A must be a positive number.",
    "test-em-props-4":                    None,
    "test-em-props-5":                    f"I must be > 0 and <= {MAX_CURR}.",
    "test-em-props-6":                    None,
    "test-inp-type-1":                    "N must be a positive integer.",
    "test-inp-type-2":                    "M must be a positive integer.",
    "test-inp-type-3":                    "K must be an integer between 1 and M (inclusive).",
    "test-inp-type-4":                    "V must be a positive number.",
    "test-inp-type-5":                    "l must be > 0 and <= the cube root of V (1.0).",
    "test-sys-setup-1":                   "K must be an integer between 1 and M (inclusive).",
    "test-sys-setup-2":                   "K must be an integer between 1 and M (inclusive).",
    "test-sys-setup-3":                   "M must be a positive integer.",
    "test-sys-setup-4":                   "V must be a positive number.",
    "test-sys-setup-5":                   "The z component (third element) of t must be positive.",
    "test-sys-setup-6":                   "l must be > 0 and <= the cube root of V (1.0).",
}


@pytest.mark.parametrize("case_id,params", list(CASES.items()))
def test_validate_input_params(case_id, params):
    expected = EXPECTED[case_id]
    if expected is None:
        res = validate_params(params)

        np.testing.assert_allclose(res.t, np.array(params["t"], dtype=float))
        np.testing.assert_allclose(res.m_t, np.array(params["m_t"], dtype=float))
    else:
        with pytest.raises(ValueError) as exc:
            validate_params(params)
        assert expected in str(exc.value), f"{case_id}: expected '{expected}', got '{exc.value}'"