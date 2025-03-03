#!/usr/bin/env python3
import json
import os
import zipfile

import numpy as np

import pyrte_rrtmgp.pyrte_rrtmgp as py


from typing import Any


def test_rrtmgp_compute_Planck_source(request: Any) -> None:

    path = os.path.dirname(request.path)

    input_data = None
    output_data = None
    with zipfile.ZipFile(f"{path}/compute_Planck_source_test_data.zip") as myzip:
        with myzip.open("compute_Planck_source_input.json") as input_data_file:
            input_data = json.load(input_data_file)
        with myzip.open("compute_Planck_source_output.json") as output_data_file:
            output_data = json.load(output_data_file)

    for key in input_data:
        values = input_data[key]
        if isinstance(values, list):
            if isinstance(values[0], int):
                values = np.array(values, dtype=np.int32)
            else:
                values = np.array(values, dtype=np.float64)
            input_data[key] = values

    args = list(input_data.values())

    py.rrtmgp_compute_Planck_source(*args)

    for key in output_data:
        assert np.allclose(output_data[key] - input_data[key], 0.0)
