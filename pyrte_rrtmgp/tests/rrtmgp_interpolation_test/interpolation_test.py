#!/usr/bin/env python3
import json
import os
import zipfile

import numpy as np

import pyrte_rrtmgp.pyrte_rrtmgp as py

from typing import Any


def load_json_file(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return json.load(f)


def test_rrtmgp_interpolation(request: Any) -> None:
    path = os.path.dirname(request.path)

    input_data = None
    output_data = None
    with zipfile.ZipFile(f"{path}/interpolation_test_data.zip") as myzip:
        with myzip.open("input.json") as input_data_file:
            input_data = json.load(input_data_file)
        with myzip.open("output.json") as output_data_file:
            output_data = json.load(output_data_file)

    assert input_data is not None and output_data is not None

    for key in input_data:
        values = input_data[key]
        if isinstance(values, list):
            if isinstance(values[0], int):
                values = np.array(values, dtype=np.int32)
            else:
                values = np.array(values, dtype=np.float64)
            input_data[key] = values

    args = list(input_data.values())

    py.rrtmgp_interpolation(*args)

    for key in output_data:
        if key in ["col_mix", "tropo"]:
            continue
        assert np.allclose(np.array(output_data[key]) - np.array(input_data[key]), 0.0)
