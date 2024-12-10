#!/usr/bin/env python3
import json
import numpy as np
import pyrte_rrtmgp.pyrte_rrtmgp as py
import os

def test_lw_solver(request):
    path = os.path.dirname(request.path)
    with open(f'{path}/lw_solver_input.json', 'r') as f:
        data = json.load(f)

    for key in data:
        values = data[key]
        if isinstance(values, list):
            values = np.array(values)
            data[key] = values

    args = list(data.values())

    py.rte_lw_solver_noscat(*args)

    test_data = np.load(f"{path}/lw_solver_output.npy")

    assert(np.allclose(test_data - data['broadband_up'], 0.))