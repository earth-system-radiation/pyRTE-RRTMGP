#!/usr/bin/env python3
import json
import os

import numpy as np

import pyrte_rrtmgp.pyrte_rrtmgp as py

from typing import Any


def load_json_file(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return json.load(f)

def test_lw_solver(request: Any) -> None:
    path = os.path.dirname(request.path)
    input_data = load_json_file(f"{path}/sw_2stream_input.json")

    for key in input_data:
        values = input_data[key]
        if isinstance(values, list):
            values = np.array(values)
            input_data[key] = values

    args = list(input_data.values())

    py.rte_sw_solver_2stream(*args)

    output_data = load_json_file(f"{path}/sw_2stream_output.json")

    for key in output_data:
        assert np.allclose(output_data[key] - input_data[key], 0.0)
