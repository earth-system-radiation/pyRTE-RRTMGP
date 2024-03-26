import xarray as xr
import sys
import numpy as np

rte_rrtmgp_dir = "/Users/josue/Documents/makepath/rte-python/rrtmgp-data"
clear_sky_example_files = f"{rte_rrtmgp_dir}/examples/rfmip-clear-sky/inputs"

rfmip = xr.load_dataset(
    f"{clear_sky_example_files}/multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc"
)

lw_gas_coeffs = xr.load_dataset(f"{rte_rrtmgp_dir}/rrtmgp-gas-lw-g256.nc")

rfmip = rfmip.sel(expt=0) # only one experiment
kdist = lw_gas_coeffs

flxdn_file = xr.load_dataset(
    f"{clear_sky_example_files}/rsd_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc"
)
flxup_file = xr.load_dataset(
    f"{clear_sky_example_files}/rsu_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc"
)


pres_layers = rfmip["pres_layer"]["layer"]
top_at_1 = pres_layers[0] < pres_layers[-1]

# RRTMGP won't run with pressure less than its minimum. so we add a small value to the minimum pressure
press_min = rfmip["pres_level"].min()
min_index = rfmip["pres_level"].argmin()
rfmip["pres_level"][min_index] = press_min + sys.float_info.epsilon

kdist_gas_names = [n.decode().strip() for n in kdist["gas_names"].values]

rfmip_vars = list(rfmip.keys())

gas_names = {n: n+"_GM" for n in kdist_gas_names if n + "_GM" in rfmip_vars}


# There is more than one config, need to work on setting the conditions for that
gas_names.update({
    "co": "carbon_monoxide_GM",
    "ch4": "methane_GM",
    "o2": "oxygen_GM",
    "n2o": "nitrous_oxide_GM",
    "n2": "nitrogen_GM",
    "co2": "carbon_dioxide_GM",
    "ccl4": "carbon_tetrachloride_GM",
    "cfc22": "hcfc22_GM",
    "h2o": "water_vapor",
    "o3": "ozone",
})

sfc_emis = rfmip["surface_emissivity"]
sfc_t = rfmip["surface_temperature"]

press_ref_log = np.log(kdist["press_ref"]).values
temp_ref = kdist["temp_ref"].values

ncol = len(rfmip["site"])
nlay = len(rfmip["layer"])
nflav = len(kdist["bnd"])

# build the gas array
# TODO: need to add dry air
col_gas = []
for gas in gas_names.values():
    gas_values = rfmip[gas].values
    if gas_values.ndim == 0:
        gas_values = np.full((ncol, nlay), gas_values) # expand the gas to all columns and layers
    col_gas.append(gas_values)
col_gas = np.stack(col_gas, axis=-1)

# start on 1 as we ignore dry air that is 0
vmr_idx = [i for i, g in enumerate(kdist_gas_names, 1) if g in gas_names]

# dry air is missing
vmr_ref = kdist["vmr_ref"].sel(absorber_ext=vmr_idx)

from pyrte_rrtmgp.pyrte_rrtmgp import rrtmgp_interpolation

# outputs
jtemp = np.ndarray([ncol, nlay], dtype=np.int32)
fmajor = np.ndarray([2, 2, 2, ncol, nlay, nflav], dtype=np.float64)
fminor = np.ndarray([2, 2, ncol, nlay, nflav], dtype=np.float64)
col_mix = np.ndarray([2, ncol, nlay, nflav], dtype=np.float64)
tropo = np.ndarray([ncol, nlay], dtype=np.int32)
jeta = np.ndarray([2, ncol, nlay, nflav], dtype=np.int32)
jpress = np.ndarray([ncol, nlay], dtype=np.int32)

interp_kwargs = dict(
    ncol = ncol,
    nlay = nlay,
    ngas = len(gas_names),
    nflav = nflav,
    neta = len(kdist["mixing_fraction"]),
    npres = len(kdist["pressure"]),
    ntemp = len(kdist["temperature"]),
    flavor = kdist["bnd_limits_gpt"].values.T, # kdist["bnd_limits_wavenumber"]
    press_ref_log = press_ref_log,
    temp_ref = kdist["temp_ref"].values,
    press_ref_log_delta = round((press_ref_log.min() - press_ref_log.max()) / (len(press_ref_log) - 1), 9),
    temp_ref_min = temp_ref.min(),
    temp_ref_delta = (temp_ref.max() - temp_ref.min()) / (len(temp_ref) - 1),
    press_ref_trop_log = kdist["press_ref_trop"].values.item(),
    vmr_ref = vmr_ref.values.transpose(2, 1, 0),
    play = rfmip["pres_layer"].values,
    tlay = rfmip["temp_layer"].values,
    col_gas = col_gas, # uses ngas dim(ncol,nlay,0:ngas)
    jtemp = jtemp,
    fmajor = fmajor,
    fminor = fminor,
    col_mix = col_mix,
    tropo = tropo,
    jeta = jeta,
    jpress = jpress
)

rrtmgp_interpolation(*list(interp_kwargs.values()))

print(tropo)
