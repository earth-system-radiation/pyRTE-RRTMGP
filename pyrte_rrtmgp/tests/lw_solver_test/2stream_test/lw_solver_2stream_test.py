#!/usr/bin/env python3
import json
import os
import zipfile

import numpy as np

import pyrte_rrtmgp.pyrte_rrtmgp as py


def test_lw_solver(request):
    path = os.path.dirname(request.path)

    # Unzip data
    input_data = None
    output_data = None
    with zipfile.ZipFile(f"{path}/lw_2stream_test_data.zip") as myzip:
        with myzip.open("lw_2stream_input.json") as input_data_file:
            input_data = json.load(input_data_file)
        with myzip.open("lw_2stream_output.json") as output_data_file:
            output_data = json.load(output_data_file)

    assert input_data is not None and output_data is not None

    for key in input_data:
        values = input_data[key]
        if isinstance(values, list):
            values = np.array(values)
            input_data[key] = values

    args = list(input_data.values())

    py.rte_lw_solver_2stream(*args)

    for key in output_data:
        assert np.allclose(output_data[key] - input_data[key], 0.0)
