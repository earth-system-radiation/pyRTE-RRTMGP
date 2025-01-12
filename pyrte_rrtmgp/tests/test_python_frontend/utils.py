import numpy as np
import xarray as xr


def convert_args_arrays(input_args, arrays_dtypes=[np.float64, np.float32]):
    args_to_test = []
    for dtype in arrays_dtypes:
        args = []
        for item in input_args:
            if isinstance(item, np.ndarray) and item.dtype in arrays_dtypes:
                output_item = item.astype(dtype)
            else:
                output_item = item
            args.append(output_item)
    args_to_test.append(args)
    args = []
    for item in input_args:
        if isinstance(item, np.ndarray) and item.dtype in arrays_dtypes:
            output_item = xr.DataArray(item)
        else:
            output_item = item
        args.append(output_item)
    args_to_test.append(args)
    return args_to_test
